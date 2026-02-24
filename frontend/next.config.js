const { PHASE_DEVELOPMENT_SERVER } = require("next/constants");

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
