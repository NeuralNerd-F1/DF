'use client';

import dynamic from 'next/dynamic';
import { useMemo, useState } from 'react';

import { EmptyState } from '@/components/ds/EmptyState';
import type { DriverSummary, StyleSpaceResponse, UnknownFingerprintResponse } from '@/lib/api/types';
import { hasWebGL } from '@/lib/style-space/hasWebGL';
import { joinStyleWithDrivers, nearestNeighbours, teammatePairs } from '@/lib/style-space/styleSpace';

import { DriverSidebar } from './DriverSidebar';
import { DriverTooltip } from './DriverTooltip';
import { SeasonFilter } from './SeasonFilter';
import { StyleSpace2D } from './StyleSpace2D';
import { StyleSpaceLegend } from './StyleSpaceLegend';
import { TeammateLinesToggle } from './TeammateLinesToggle';
import { UnknownProjectionPanel } from './UnknownProjectionPanel';

// Three.js scene is dynamically imported (ssr:false) so `three` never ships to the SSR / 2D-fallback
// path (div #68). A WebGL init failure inside it calls onInitError → we silently switch to 2D.
const StyleSpaceScene = dynamic(() => import('./StyleSpaceScene').then((m) => m.StyleSpaceScene), {
  ssr: false,
});

// Screen 5 root (master §10.6). Joins /stylespace × /drivers, derives NN + teammate links, chooses
// the Three.js scene vs the D3 2D fallback via hasWebGL(), and hosts the tooltip / sidebar / toggles.
export function StyleSpaceScreen({
  initialStyle,
  initialDrivers,
  forceMode,
}: {
  initialStyle: StyleSpaceResponse;
  initialDrivers: DriverSummary[];
  forceMode?: 'webgl' | '2d'; // test hook; production uses hasWebGL()
}) {
  const nodes = useMemo(
    () => joinStyleWithDrivers(initialStyle, initialDrivers),
    [initialStyle, initialDrivers],
  );
  const teammates = useMemo(() => teammatePairs(nodes), [nodes]);

  const [mode, setMode] = useState<'webgl' | '2d'>(() => forceMode ?? (hasWebGL() ? 'webgl' : '2d'));
  const [hovered, setHovered] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [showTeammates, setShowTeammates] = useState(false);
  const [unknownVec, setUnknownVec] = useState<number[] | null>(null);

  const hoveredNode = nodes.find((n) => n.code === hovered) ?? null;
  const selectedNode = nodes.find((n) => n.code === selected) ?? null;
  const hoverNN = hovered ? nearestNeighbours(hovered, nodes) : [];

  if (nodes.length === 0) {
    return (
      <main className="mx-auto max-w-5xl p-6" data-testid="style-space-screen" data-mode="empty">
        <h1 className="mb-4 font-mono text-lg uppercase tracking-wide text-accent">Style Space</h1>
        <EmptyState message="No style space computed yet." />
      </main>
    );
  }

  function onProjected(r: UnknownFingerprintResponse | null) {
    setUnknownVec(r?.fingerprint_vector ?? null);
  }

  return (
    <main className="mx-auto max-w-6xl p-6" data-testid="style-space-screen" data-mode={mode}>
      <header className="mb-2 flex flex-wrap items-center gap-3">
        <h1 className="font-mono text-lg uppercase tracking-wide text-accent">Style Space</h1>
        <TeammateLinesToggle value={showTeammates} onChange={setShowTeammates} />
        <SeasonFilter />
      </header>

      {/* What the dots are + where they come from — the scene is otherwise unlabelled coloured dots. */}
      <p
        className="mb-4 max-w-3xl font-mono text-[11px] leading-relaxed text-text-secondary"
        data-testid="style-space-about"
      >
        Each point is one driver&apos;s <span className="text-text-primary">behavioural fingerprint</span>{' '}
        — their per-lap 28-feature vector (throttle, braking, cornering, tyre usage…) averaged over the
        season, then projected from 28 dimensions down to this map by UMAP. Points that sit close
        together drive alike; <span className="text-text-primary">colour marks the constructor</span>.
        Hover or pick a driver for its nearest neighbours, drag to orbit, scroll to zoom. A projected
        unknown driver appears as a drifting white sphere. Showing{' '}
        <span className="text-text-primary">{nodes.length}</span> drivers from the loaded dataset.
      </p>

      <div className="grid gap-4 lg:grid-cols-[1fr_18rem]">
        <div className="relative rounded-lg border border-border bg-card p-2">
          {mode === 'webgl' ? (
            <StyleSpaceScene
              nodes={nodes}
              teammates={teammates}
              showTeammates={showTeammates}
              selected={selected}
              onSelect={setSelected}
              onHover={setHovered}
              unknownVector={unknownVec}
              onInitError={() => setMode('2d')}
            />
          ) : (
            <StyleSpace2D
              nodes={nodes}
              teammates={teammates}
              showTeammates={showTeammates}
              selected={selected}
              onSelect={setSelected}
              onHover={setHovered}
            />
          )}
          {hoveredNode ? (
            <div className="absolute right-2 top-2" data-testid="tooltip-anchor">
              <DriverTooltip node={hoveredNode} neighbours={hoverNN} />
            </div>
          ) : null}
        </div>

        <div className="space-y-4">
          {selectedNode ? <DriverSidebar node={selectedNode} onClose={() => setSelected(null)} /> : null}
          <StyleSpaceLegend
            nodes={nodes}
            selected={selected}
            onHover={setHovered}
            onSelect={setSelected}
          />
          <UnknownProjectionPanel onProjected={onProjected} />
        </div>
      </div>
    </main>
  );
}
