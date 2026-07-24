/** @type {import('tailwindcss').Config} */
// APEX design system — master §10.1 (F16 spec §4.1 / §5). This file is the SINGLE SOURCE OF
// TRUTH for design tokens: no colour/font literal belongs in a component. Dark theme only.
module.exports = {
  darkMode: 'class',
  content: [
    './app/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './lib/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        // §10.1 verbatim tokens
        background: '#0D0D0D', // near-black page background
        accent: {
          DEFAULT: '#E8002D', // F1 red — headings, active states, alert indicators
          deep: '#C0392B', // deep red — SHAP force values, warnings
        },
        text: {
          primary: '#F0F0F0', // near-white body text
          secondary: '#888888', // mid-grey labels / metadata
        },
        card: '#1A1A1A', // card surface (lighter than background)
        border: '#2B2B2B', // subtle separation
        // Confidence-state colours. Master §10.4 names green/amber/red only (no hex) → F16-defined
        // (spec div #46). The confidence bar colours off confidence_state, NOT a numeric re-threshold.
        confidence: {
          high: '#21C08B', // green  → HIGH
          medium: '#E8A33D', // amber → MEDIUM
          low: '#E8002D', // red    → LOW_CONFIDENCE / AMBIGUOUS
        },
        // F17 (master §10.4 SHAP force / §10.5 delta radar). Master names red "toward"/positive and
        // blue "away"/negative but gives no blue hex → F17-defined (div #53). Red reuses the accent.
        delta: {
          up: '#E8002D', // positive / toward prediction / driver-1 higher
          down: '#3B82F6', // negative / away / driver-2 higher (blue)
        },
      },
      fontFamily: {
        // Wired to next/font CSS variables in app/layout.tsx.
        sans: ['var(--font-inter)', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['var(--font-jetbrains-mono)', 'ui-monospace', 'monospace'],
      },
    },
  },
  plugins: [],
};
