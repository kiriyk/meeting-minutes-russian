fn main() {
    // LAPACK's Fortran ABI is supplied by Accelerate itself on macOS.
    // netlib-src/system additionally links gfortran, which is unnecessary here.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
