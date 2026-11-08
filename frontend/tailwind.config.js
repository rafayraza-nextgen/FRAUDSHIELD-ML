/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: "class",
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        surface: "var(--surface)",
        "surface-card": "var(--surface-card)",
        "surface-border": "var(--surface-border)",
        brand: {
          50: "#EFF6FF",
          100: "#DBEAFE",
          200: "#BFDBFE",
          500: "#3B82F6",
          600: "#2563EB",
          700: "#1D4ED8",
          800: "#1E40AF",
          900: "#1E3A8A",
        },
        cyber: {
          cyan: "#06B6D4",
          purple: "#8B5CF6",
          neon: "#00F0FF",
        },
        risk: {
          low: "#10B981",
          moderate: "#F59E0B",
          high: "#EF4444",
        },
      },
      boxShadow: {
        cyber: "0 0 20px -5px rgba(6, 182, 212, 0.3)",
        "cyber-purple": "0 0 20px -5px rgba(139, 92, 246, 0.3)",
        "cyber-danger": "0 0 20px -5px rgba(239, 68, 68, 0.3)",
      },
    },
  },
  plugins: [],
};
