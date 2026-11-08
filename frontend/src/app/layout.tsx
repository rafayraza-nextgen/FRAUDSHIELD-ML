import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "F.R.A.U.D.S.H.I.E.L.D. — Financial Risk Assessment & Unauthorized Data Shield",
  description:
    "AI-powered enterprise fraud detection engine featuring real-time risk scoring, SHAP explainability, and multi-model benchmark evaluation.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  const storedTheme = localStorage.getItem('fraudshield-theme');
                  const supportDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
                  if (storedTheme === 'dark' || (!storedTheme && supportDarkMode)) {
                    document.documentElement.classList.add('dark');
                  } else {
                    document.documentElement.classList.remove('dark');
                  }
                } catch (e) {}
              })();
            `,
          }}
        />
      </head>
      <body className="cyber-grid antialiased transition-colors duration-200">
        {children}
      </body>
    </html>
  );
}
