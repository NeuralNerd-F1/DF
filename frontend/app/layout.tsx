import type { Metadata } from 'next';
import { Barlow_Condensed, Inter, JetBrains_Mono } from 'next/font/google';
import type { ReactNode } from 'react';

import { NavTabs } from '@/components/chrome/NavTabs';
import { SystemStatus } from '@/components/chrome/SystemStatus';

import './globals.css';

// Master §10.1: Inter for UI text, JetBrains Mono for telemetry values / lap numbers / error codes.
// Barlow Condensed is the display face — driver numbers, names, screen titles — set against the
// wide-tracked mono micro-labels (the reference's core typographic contrast).
const inter = Inter({ subsets: ['latin'], variable: '--font-inter', display: 'swap' });
const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
  display: 'swap',
});
const barlowCondensed = Barlow_Condensed({
  subsets: ['latin'],
  weight: ['500', '600', '700'],
  variable: '--font-display',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'APEX Driver Profiling',
  description: 'F1 driver behavioural profiling. Behavioural fingerprints built from raw telemetry.',
};

// Dark theme only (master §10.1 / §9 Styling). The `dark` class + font variables live on <html>.
export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html
      lang="en"
      className={`dark ${inter.variable} ${jetbrainsMono.variable} ${barlowCondensed.variable}`}
    >
      <body className="flex min-h-screen flex-col bg-background font-sans text-text-primary antialiased">
        <header className="flex h-14 items-center gap-8 border-b border-border px-6">
          <div className="flex items-center gap-3">
            <span className="font-display text-2xl font-bold tracking-[0.08em] text-accent">
              APEX
            </span>
            <span className="hidden border-l border-border pl-3 font-mono text-[8px] uppercase leading-[1.5] tracking-[0.22em] text-text-secondary lg:block">
              Adaptive Performance &amp;
              <br />
              Experience Profiling System
            </span>
          </div>
          <div className="flex flex-1 justify-center self-stretch">
            <NavTabs />
          </div>
          <SystemStatus />
        </header>

        <div className="flex-1">{children}</div>

        <footer className="flex flex-wrap items-center justify-between gap-2 border-t border-border px-6 py-4 font-mono text-[9px] uppercase tracking-[0.2em] text-text-secondary">
          <span>APEX, the Adaptive Performance and Experience Profiling System</span>
          <span className="flex items-center gap-4">
            <span>XGBoost classifier · UMAP projection · SHAP explanations</span>
            <span className="text-accent">Research prototype</span>
          </span>
        </footer>
      </body>
    </html>
  );
}
