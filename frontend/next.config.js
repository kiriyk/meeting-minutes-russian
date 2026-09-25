const path = require("path");
const { PHASE_DEVELOPMENT_SERVER } = require("next/constants");

const tiptapPmResolveBase = path.dirname(require.resolve("@tiptap/pm/model"));
const resolveFromTiptapPm = (pkg) =>
  require.resolve(pkg, { paths: [tiptapPmResolveBase] });

module.exports = (phase) => {
  const isDev = phase === PHASE_DEVELOPMENT_SERVER;

  /** @type {import('next').NextConfig} */
  const nextConfig = {
    reactStrictMode: false, // Disabled for BlockNote compatibility
    // Keep static export only for production build. In dev it can break chunk loading in WebView.
    ...(isDev ? {} : { output: "export" }),
    images: {
      unoptimized: true,
    },
    basePath: "",
    assetPrefix: "/",
    webpack: (config, { isServer, dev }) => {
      if (!isServer) {
        config.resolve.fallback = {
          ...config.resolve.fallback,
          fs: false,
          path: false,
          os: false,
        };

        // Keep ProseMirror single-instanced for BlockNote/Tiptap.
        config.resolve.alias = {
          ...config.resolve.alias,
          "@blocknote/core$": require.resolve("@blocknote/core"),
          "@blocknote/react$": require.resolve("@blocknote/react"),
          "@blocknote/shadcn$": require.resolve("@blocknote/shadcn"),
          "prosemirror-model": resolveFromTiptapPm("prosemirror-model"),
          "prosemirror-state": resolveFromTiptapPm("prosemirror-state"),
          "prosemirror-view": resolveFromTiptapPm("prosemirror-view"),
          "prosemirror-transform": resolveFromTiptapPm("prosemirror-transform"),
          "prosemirror-tables": resolveFromTiptapPm("prosemirror-tables"),
          "prosemirror-schema-list": resolveFromTiptapPm("prosemirror-schema-list"),
          "prosemirror-keymap": resolveFromTiptapPm("prosemirror-keymap"),
          "prosemirror-commands": resolveFromTiptapPm("prosemirror-commands"),
          "prosemirror-history": resolveFromTiptapPm("prosemirror-history"),
          "prosemirror-inputrules": resolveFromTiptapPm("prosemirror-inputrules"),
          "prosemirror-gapcursor": resolveFromTiptapPm("prosemirror-gapcursor"),
          "prosemirror-dropcursor": resolveFromTiptapPm("prosemirror-dropcursor"),
        };
      }

      // Avoid eval-based chunks in dev (WKWebView can fail with "Invalid or unexpected token").
      if (dev) {
        config.devtool = "cheap-module-source-map";
      }

      return config;
    },
  };

  return nextConfig;
};
