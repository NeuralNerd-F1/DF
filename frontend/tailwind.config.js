/** @type {import('tailwindcss').Config} */
// APEX design system — pastel reference (dissertation branch). This file is the SINGLE SOURCE
// OF TRUTH for design tokens: no colour/font literal belongs in a component. The master §10.1
// dark-only rule applies to master's SENNA system; this branch is the light pastel counterpart.
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
        // APEX Pastel reference (dissertation branch). Every token maps to a state the API
        // returns; pastel FILLS carry area, INKS carry strokes, glyphs and text — never set
        // text in a fill colour. The §10.1 dark system stays on master; the two systems share
        // no surface value and no accent hue.
        background: '#F6F5FA', // paper — cool tinted page canvas, never pure white
        card: '#FFFFFF', // surface — driver cards, panels, chart wells
        raised: '#EEECF5', // segment tiles, inactive controls, bar tracks
        border: '#DCD8E8', // hairlines, radar web, axes, gridlines
        text: {
          primary: '#2B2733', // driver names, headings, measured values
          secondary: '#6C6579', // constructor, summaries, disclaimers
          tertiary: '#9A93A8', // field labels, footer, axis ticks
        },
        // Assert — the model stating something (crimson softened to rose).
        accent: {
          DEFAULT: '#CE3A72', // assert-ink — wordmark, active tab, bar stroke, value text
          deep: '#94284F', // assert-deep — hover, pressed, selected-tile border
          fill: '#FBD5E3', // assert-fill — predicted-driver card, confidence bar fill
        },
        // Confidence-state colours off confidence_state, NOT a numeric re-threshold. Hue changes
        // at the gate: below it the system is not weakly asserting, it is REFUSING to assert —
        // so LOW/AMBIGUOUS/OOD take the epistemic purple, not a paler rose.
        confidence: {
          high: '#CE3A72', // asserting at strength (assert-ink)
          medium: '#E17A9E', // asserting mid-ramp
          low: '#7A5DC0', // LOW_CONFIDENCE / AMBIGUOUS / OOD → uncertain-ink
        },
        // Signed SHAP evidence — positive reuses the assert pair; negative is the against pair
        // (replaces the master's cyan/blue, div #53 still F17-defined).
        delta: {
          up: '#CE3A72', // toward prediction / driver-1 higher (assert-ink)
          down: '#4F69C9', // away / driver-2 higher (against-ink)
          'up-fill': '#FBD5E3', // positive contribution bar area
          'down-fill': '#D8DFFB', // negative contribution bar area
        },
        // Epistemic — the model declining to answer.
        uncertain: {
          fill: '#E4DAF8', // gated-null card
          ink: '#7A5DC0', // its border, label and icon
        },
        unavailable: '#B4AEC0', // SHAP_UNAVAILABLE, empty state, deferred seam
        // Operational — serving, but compromised.
        alert: {
          fill: '#FCE6C3', // confirmed alert band, DEGRADED, STALE_DATA
          ink: '#B07520', // alert stroke, degradation rate, PERF_WARN
        },
        watch: '#CBA45E', // single crossing — escalates by intensity, not hue
        threshold: '#8B93A6', // the tau line, baselines, self-resolved excursions
        track: '#E0DEE8', // empty bar track, classifier output rows
      },
      fontFamily: {
        // Wired to next/font CSS variables in app/layout.tsx.
        sans: ['var(--font-inter)', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['var(--font-jetbrains-mono)', 'ui-monospace', 'monospace'],
        // Condensed display face — driver numbers, names, screen titles (the reference look:
        // condensed display type set against wide-tracked mono micro-labels).
        display: ['var(--font-display)', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
      // F19 (Screen 6 §10.7): the degradation alert banner "animates down" on fire. A design token,
      // not a component literal — matches how all motion constants live outside components (div #70).
      keyframes: {
        slideDown: {
          '0%': { transform: 'translateY(-8px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
      animation: {
        'slide-down': 'slideDown 300ms ease-in-out',
      },
    },
  },
  plugins: [],
};
