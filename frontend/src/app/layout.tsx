/*
  layout.tsx — root Next.js layout.
  Wraps every page with the HTML shell, font, and metadata.
*/
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "O-RAN KPI xApp",
  description:
    "ML-based KPI Prediction xApp for O-RAN Network Monitoring — FYP",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#0f172a] text-slate-200 antialiased">
        {children}
      </body>
    </html>
  );
}
