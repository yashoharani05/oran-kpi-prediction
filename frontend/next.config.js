/** @type {import('next').NextConfig} */
const nextConfig = {
  // -----------------------------------------------------------------------
  // output: 'standalone'
  //
  // Tells Next.js to bundle everything needed to run the app into
  // .next/standalone/ — a single folder with no external node_modules.
  // This is required for the two-stage Docker build in the frontend Dockerfile.
  // Without this, the Docker runner stage cannot start the server.
  //
  // When running locally with `npm run dev`, this setting is ignored.
  // -----------------------------------------------------------------------
  output: "standalone",

  // -----------------------------------------------------------------------
  // API proxy rewrites
  //
  // Requests to /api/* are forwarded to the backend.
  //
  // BACKEND_URL:
  //   - In Docker: http://backend:8000  (service name from docker-compose.yml)
  //   - Locally:   http://localhost:8000
  //
  // The environment variable is set by:
  //   - docker-compose.yml (for Docker)
  //   - The default fallback below (for local development)
  // -----------------------------------------------------------------------
  async rewrites() {
    const backendUrl =
      process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
