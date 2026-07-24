import { describe, expect, it } from 'vitest';

import type { DriverSummary, StyleSpaceResponse } from '@/lib/api/types';

import {
  cosineSimilarity,
  joinStyleWithDrivers,
  nearestNeighbours,
  teammatePairs,
  type StyleNode,
} from './styleSpace';

const styleRow = (code: string, c2: number[], c5: number[]) => ({
  id: code,
  code,
  coords_2d: c2,
  coords_5d: c5,
  season: null,
});

describe('joinStyleWithDrivers (div #65)', () => {
  it('enriches each style row with team + radar from /drivers by code', () => {
    const style = { drivers: [styleRow('VER', [1, 2], [1, 0, 0, 0, 0])] } as unknown as StyleSpaceResponse;
    const drivers = [{ id: 'VER', code: 'VER', team: 'Red Bull', radar_values: { throttle: 0.5 } }] as unknown as DriverSummary[];
    const [node] = joinStyleWithDrivers(style, drivers);
    expect(node.team).toBe('Red Bull');
    expect(node.radar).toEqual({ throttle: 0.5 });
    expect(node.coords2d).toEqual([1, 2]);
  });

  it('keeps the scatter position but nulls team for a driver absent from /drivers (honest degrade)', () => {
    const style = { drivers: [styleRow('XXX', [3, 4], [0, 1, 0, 0, 0])] } as unknown as StyleSpaceResponse;
    const [node] = joinStyleWithDrivers(style, [] as unknown as DriverSummary[]);
    expect(node.team).toBeNull();
    expect(node.radar).toBeUndefined();
    expect(node.coords2d).toEqual([3, 4]);
  });
});

describe('cosineSimilarity', () => {
  it('is 1 for identical vectors and 0 for orthogonal', () => {
    expect(cosineSimilarity([1, 2, 3], [1, 2, 3])).toBeCloseTo(1);
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0);
  });
  it('is 0 when either vector is all-zero (no NaN)', () => {
    expect(cosineSimilarity([0, 0], [1, 1])).toBe(0);
  });
});

const nodes: StyleNode[] = [
  { code: 'VER', coords2d: [0, 0], coords5d: [1, 0, 0], team: 'Red Bull' },
  { code: 'PER', coords2d: [0, 0], coords5d: [0.9, 0.1, 0], team: 'Red Bull' },
  { code: 'HAM', coords2d: [0, 0], coords5d: [0, 1, 0], team: 'Mercedes' },
  { code: 'LEC', coords2d: [0, 0], coords5d: [0, 0, 1], team: 'Ferrari' },
];

describe('nearestNeighbours (div #66)', () => {
  it('returns the top-k by coords_5d cosine similarity, self excluded', () => {
    const nn = nearestNeighbours('VER', nodes, 2);
    expect(nn.map((n) => n.code)).toEqual(['PER', expect.any(String)]);
    expect(nn[0].code).toBe('PER');
    expect(nn).toHaveLength(2);
  });
});

describe('teammatePairs (master §10.6)', () => {
  it('links same-team drivers only, with an intra-team similarity', () => {
    const links = teammatePairs(nodes);
    expect(links).toHaveLength(1); // only VER-PER share a team
    expect([links[0].a, links[0].b].sort()).toEqual(['PER', 'VER']);
    expect(links[0].similarity).toBeGreaterThan(0.9);
  });
  it('produces no link for a lone driver on a team', () => {
    const lone = nodes.filter((n) => n.team !== 'Red Bull'); // HAM + LEC, different teams
    expect(teammatePairs(lone)).toHaveLength(0);
  });
});
