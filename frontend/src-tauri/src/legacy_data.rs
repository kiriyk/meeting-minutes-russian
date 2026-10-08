//! One-time move of app data from the upstream bundle identifier.
//!
//! The fork ships as `ru.kiriyk.meetily` so it can be installed and updated
//! independently of upstream Meetily (`com.meetily.ai`). Everything resolved
//! through `app_data_dir()` (database, models, preferences, templates) lives in
//! a directory named after the identifier, so without this an upgraded install
//! would start empty.
use log::{info, warn};
use std::fs;
use std::io;
use std::path::Path;

/// Bundle identifier used before the fork had its own.
pub const LEGACY_IDENTIFIER: &str = "com.meetily.ai";

/// File whose presence marks a data directory as already in use.
const DATABASE_FILE: &str = "meeting_minutes.sqlite";

/// Moves entries from `legacy_dir` into `app_dir` unless `app_dir` already has a
/// database. Entries already present in `app_dir` are left untouched. Each entry
/// is renamed (instant on one volume) and copied only if renaming fails.
/// Returns the number of entries moved.
pub fn migrate(legacy_dir: &Path, app_dir: &Path) -> io::Result<usize> {
    if !legacy_dir.is_dir() || app_dir.join(DATABASE_FILE).exists() {
        return Ok(0);
    }
    fs::create_dir_all(app_dir)?;

    let mut moved = 0;
    for entry in fs::read_dir(legacy_dir)? {
        let entry = entry?;
        let target = app_dir.join(entry.file_name());
        if target.exists() {
            continue;
        }
        if fs::rename(entry.path(), &target).is_err() {
            copy_recursive(&entry.path(), &target)?;
            remove_recursive(&entry.path())?;
        }
        moved += 1;
    }

    // Leave the legacy directory only if something in it was not moved.
    let _ = fs::remove_dir(legacy_dir);
    Ok(moved)
}

/// Runs [`migrate`] for the running app; failures are logged, never fatal.
pub fn migrate_app_data<R: tauri::Runtime>(app: &tauri::AppHandle<R>) {
    use tauri::Manager;
    let Ok(app_dir) = app.path().app_data_dir() else {
        return;
    };
    let Some(legacy_dir) = app_dir.parent().map(|p| p.join(LEGACY_IDENTIFIER)) else {
        return;
    };
    if legacy_dir == app_dir {
        return;
    }
    match migrate(&legacy_dir, &app_dir) {
        Ok(0) => {}
        Ok(moved) => info!(
            "Moved {} entries of app data from {} to {}",
            moved,
            legacy_dir.display(),
            app_dir.display()
        ),
        Err(e) => warn!(
            "Failed to move app data from {} to {}: {}",
            legacy_dir.display(),
            app_dir.display(),
            e
        ),
    }
}

fn copy_recursive(from: &Path, to: &Path) -> io::Result<()> {
    if from.is_dir() {
        fs::create_dir_all(to)?;
        for entry in fs::read_dir(from)? {
            let entry = entry?;
            copy_recursive(&entry.path(), &to.join(entry.file_name()))?;
        }
        Ok(())
    } else {
        fs::copy(from, to).map(|_| ())
    }
}

fn remove_recursive(path: &Path) -> io::Result<()> {
    if path.is_dir() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, content: &str) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, content).unwrap();
    }

    #[test]
    fn moves_database_models_and_preferences_into_a_new_app_dir() {
        let root = tempfile::tempdir().unwrap();
        let legacy = root.path().join(LEGACY_IDENTIFIER);
        let app = root.path().join("ru.kiriyk.meetily");
        write(&legacy.join(DATABASE_FILE), "db");
        write(&legacy.join("models/gigaam/encoder.onnx"), "weights");
        write(&legacy.join("preferences.json"), "{}");

        assert_eq!(migrate(&legacy, &app).unwrap(), 3);

        assert_eq!(fs::read_to_string(app.join(DATABASE_FILE)).unwrap(), "db");
        assert_eq!(fs::read_to_string(app.join("models/gigaam/encoder.onnx")).unwrap(), "weights");
        assert_eq!(fs::read_to_string(app.join("preferences.json")).unwrap(), "{}");
        assert!(!legacy.exists(), "an emptied legacy dir is removed");
    }

    #[test]
    fn keeps_entries_the_new_app_dir_already_has() {
        let root = tempfile::tempdir().unwrap();
        let legacy = root.path().join(LEGACY_IDENTIFIER);
        let app = root.path().join("ru.kiriyk.meetily");
        write(&legacy.join(DATABASE_FILE), "old db");
        write(&legacy.join("analytics.json"), "old");
        // Created by the new app before setup ran, e.g. a webview cache.
        write(&app.join("analytics.json"), "new");

        assert_eq!(migrate(&legacy, &app).unwrap(), 1);

        assert_eq!(fs::read_to_string(app.join(DATABASE_FILE)).unwrap(), "old db");
        assert_eq!(fs::read_to_string(app.join("analytics.json")).unwrap(), "new");
        assert_eq!(fs::read_to_string(legacy.join("analytics.json")).unwrap(), "old");
    }

    #[test]
    fn does_nothing_once_the_new_app_dir_has_a_database() {
        let root = tempfile::tempdir().unwrap();
        let legacy = root.path().join(LEGACY_IDENTIFIER);
        let app = root.path().join("ru.kiriyk.meetily");
        write(&legacy.join(DATABASE_FILE), "old db");
        write(&app.join(DATABASE_FILE), "new db");

        assert_eq!(migrate(&legacy, &app).unwrap(), 0);
        assert_eq!(fs::read_to_string(app.join(DATABASE_FILE)).unwrap(), "new db");
        assert!(legacy.join(DATABASE_FILE).exists());
    }

    #[test]
    fn does_nothing_without_a_legacy_dir() {
        let root = tempfile::tempdir().unwrap();
        let app = root.path().join("ru.kiriyk.meetily");
        assert_eq!(migrate(&root.path().join(LEGACY_IDENTIFIER), &app).unwrap(), 0);
        assert!(!app.exists());
    }

    #[test]
    fn copies_when_rename_is_impossible() {
        let root = tempfile::tempdir().unwrap();
        let from = root.path().join("from");
        write(&from.join("models/a.bin"), "a");
        write(&from.join("b.json"), "b");
        let to = root.path().join("to");
        copy_recursive(&from, &to).unwrap();
        assert_eq!(fs::read_to_string(to.join("models/a.bin")).unwrap(), "a");
        assert_eq!(fs::read_to_string(to.join("b.json")).unwrap(), "b");
    }
}
