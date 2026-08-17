import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { confidenceTone } from './confidence';

const ROOT = process.cwd();

function walk(dir: string): string[] {
  const out: string[] = [];
  for (const entry of readdirSync(join(ROOT, dir))) {
    if (entry === 'node_modules' || entry === '.next') continue;
    const rel = join(dir, entry);
    if (statSync(join(ROOT, rel)).isDirectory()) {
      out.push(...walk(rel));
    } else if (/\.(ts|tsx)$/.test(entry) && !/\.test\.tsx?$/.test(entry)) {
      out.push(rel); // source files only — never the tests themselves
    }
  }
  return out;
}

// Scan CODE, not the comments documenting the rule (a comment saying "not a spinner" is not a
// violation). Strip block + line comments first.
function codeOf(rel: string): string {
  return readFileSync(join(ROOT, rel), 'utf8')
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/(^|[^:])\/\/[^\n]*/g, '$1');
}

describe('design-system guards (F16 §7.1/§7.3)', () => {
  it('tailwind.config.js carries every pastel-reference token', () => {
    const cfg = readFileSync(join(ROOT, 'tailwind.config.js'), 'utf8');
    for (const hex of [
      '#F6F5FA', // paper bg
      '#FFFFFF', // surface
      '#EEECF5', // raised
      '#DCD8E8', // border
      '#2B2733', // text
      '#6C6579', // text-2
      '#9A93A8', // text-3
      '#CE3A72', // assert-ink
      '#94284F', // assert-deep
      '#FBD5E3', // assert-fill
      '#4F69C9', // against-ink
      '#D8DFFB', // against-fill
      '#7A5DC0', // uncertain-ink
      '#E4DAF8', // uncertain-fill
      '#B4AEC0', // unavailable
      '#FCE6C3', // alert-fill
      '#B07520', // alert-ink
      '#CBA45E', // watch-ink
      '#8B93A6', // threshold
      '#E0DEE8', // track
    ]) {
      expect(cfg).toContain(hex);
    }
    expect(cfg).toContain('font-inter');
    expect(cfg).toContain('font-jetbrains-mono');
  });

  it('no component or page re-derives the confidence gate (no 0.65 / 0.85 literal in code)', () => {
    for (const f of [...walk('components'), ...walk('app')]) {
      expect(codeOf(f), `${f} must not hardcode the gate`).not.toMatch(/0\.65|0\.85/);
    }
  });

  it('uses skeletons, never spinners', () => {
    for (const f of [...walk('components'), ...walk('app'), ...walk('lib')]) {
      expect(codeOf(f).toLowerCase(), `${f} must not contain a spinner`).not.toContain('spinner');
    }
  });
});

describe('confidenceTone maps every state', () => {
  it('confident -> high/medium, uncertainty -> low, unknown -> low', () => {
    expect(confidenceTone('HIGH')).toBe('high');
    expect(confidenceTone('MEDIUM')).toBe('medium');
    expect(confidenceTone('LOW_CONFIDENCE')).toBe('low');
    expect(confidenceTone('AMBIGUOUS')).toBe('low');
    expect(confidenceTone('OUT_OF_DISTRIBUTION')).toBe('low');
    expect(confidenceTone(null)).toBe('low');
    expect(confidenceTone(undefined)).toBe('low');
  });
});
