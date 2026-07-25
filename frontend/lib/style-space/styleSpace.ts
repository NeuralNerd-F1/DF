// Pure style-space helpers (master §10.6). The screen joins /stylespace with /drivers (team +
// mini-radar) and derives nearest-neighbours + teammate links CLIENT-SIDE from coords_5d, because
// /stylespace carries neither team nor neighbours (div #65/#66). All functions are deterministic and
// unit-tested; no Math.random, no I/O.
import type { DriverSummary, StyleSpaceResponse } from '@/lib/api/types';

export type StyleNode = {
  code: string;
  coords2d: [number, number];
  coords5d: number[];
  team: string | null;
  radar?: Record<string, number>;
};

function pad2(a: number[]): [number, number] {
  return [a[0] ?? 0, a[1] ?? 0];
}

// Join /stylespace rows with /drivers by `code`. A driver absent from /drivers keeps its scatter
// position but gets a null team (→ neutral colour) and no radar — honest degrade, never fabricated.
export function joinStyleWithDrivers(
  style: StyleSpaceResponse,
  drivers: DriverSummary[],
): StyleNode[] {
  const byCode = new Map(drivers.map((d) => [d.code, d]));
  return (style.drivers ?? []).map((s) => {
    const d = byCode.get(s.code);
    return {
      code: s.code,
      coords2d: pad2(s.coords_2d ?? []),
      coords5d: s.coords_5d ?? [],
      team: d?.team ?? null,
      radar: d?.radar_values ?? undefined,
    };
  });
}

export function cosineSimilarity(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < n; i++) {
    const av = a[i] ?? 0;
    const bv = b[i] ?? 0;
    dot += av * bv;
    na += av * av;
    nb += bv * bv;
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

export type Neighbour = { code: string; similarity: number };

// Top-k known-driver neighbours by cosine similarity on coords_5d (div #66).
export function nearestNeighbours(code: string, nodes: StyleNode[], k = 3): Neighbour[] {
  const self = nodes.find((n) => n.code === code);
  if (!self || self.coords5d.length === 0) return [];
  return nodes
    .filter((n) => n.code !== code && n.coords5d.length > 0)
    .map((n) => ({ code: n.code, similarity: cosineSimilarity(self.coords5d, n.coords5d) }))
    .sort((x, y) => y.similarity - x.similarity)
    .slice(0, k);
}

export type TeammateLink = { a: string; b: string; similarity: number };

// Connecting links between same-team drivers; thickness ∝ intra-team similarity (master §10.6).
export function teammatePairs(nodes: StyleNode[]): TeammateLink[] {
  const links: TeammateLink[] = [];
  for (let i = 0; i < nodes.length; i += 1) {
    for (let j = i + 1; j < nodes.length; j += 1) {
      const a = nodes[i];
      const b = nodes[j];
      if (a && b && a.team && b.team && a.team === b.team) {
        links.push({ a: a.code, b: b.code, similarity: cosineSimilarity(a.coords5d, b.coords5d) });
      }
    }
  }
  return links;
}

// 3 of 5 UMAP dims for the Three.js scatter (master §10.6); pads short vectors with 0.
export function pos3(coords5d: number[]): [number, number, number] {
  return [coords5d[0] ?? 0, coords5d[1] ?? 0, coords5d[2] ?? 0];
}
