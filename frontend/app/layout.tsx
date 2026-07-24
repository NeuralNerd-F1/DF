import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import Link from 'next/link';
import type { ReactNode } from 'react';

import './globals.css';

// Master §10.1: Inter for UI text, JetBrains Mono for telemetry values / lap numbers / error codes.
const inter = Inter({ subsets: ['latin'], variable: '--font-inter', display: 'swap' });
const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'APEX — Driver Profiling',
  description: 'F1 driver behavioural profiling — Command Centre & Driver Profile.',
};

// Dark theme only (master §10.1 / §9 Styling). The `dark` class + font variables live on <html>.
export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`dark ${inter.variable} ${jetbrainsMono.variable}`}>
      <body className="min-h-screen bg-background font-sans text-text-primary antialiased">
        <nav className="flex gap-4 border-b border-border px-6 py-3 font-mono text-xs text-text-secondary">
          <Link href="/" className="transition-colors hover:text-accent">
            Command Centre
          </Link>
          <Link href="/ingest" className="transition-colors hover:text-accent">
            Ingestion &amp; Prediction
          </Link>
          <Link href="/compare" className="transition-colors hover:text-accent">
            Comparison
          </Link>
          <Link href="/style-space" className="transition-colors hover:text-accent">
            Style Space
          </Link>
          <Link href="/replay" className="transition-colors hover:text-accent">
            Replay Monitor
          </Link>
        </nav>
        {children}
      </body>
    </html>
  );
}
