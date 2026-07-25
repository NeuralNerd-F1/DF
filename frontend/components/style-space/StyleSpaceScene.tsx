'use client';

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { useEffect, useRef } from 'react';

import { teamColour } from '@/lib/design/teams';
import { CAMERA_ROTATE_DEG_PER_SEC, DURATION } from '@/lib/design/animation';
import { pos3, type StyleNode, type TeammateLink } from '@/lib/style-space/styleSpace';

// Three.js 3D scatter (master §10.6): team-coloured glowing spheres at 3 of 5 UMAP dims, an
// auto-rotating orbit camera (0.2°/s), hover, click-to-select, teammate lines, and an unknown-driver
// white sphere. ALL WebGL setup is wrapped in try/catch → onInitError, so a WebGL failure (context
// lost / init throw) silently degrades to the D3 2D fallback (master §10.6 "no error", a §6 failure
// mode). The scene is dynamically imported (ssr:false) so `three` never ships to the fallback path.
type Props = {
  nodes: StyleNode[];
  teammates: TeammateLink[];
  showTeammates: boolean;
  selected: string | null;
  onSelect: (code: string) => void;
  onHover: (code: string | null) => void;
  unknownVector?: number[] | null;
  onInitError?: () => void;
};

// Scale raw UMAP coords to a tidy cube so the camera framing is stable regardless of coord range.
function makeScaler(nodes: StyleNode[]): (c: number[]) => THREE.Vector3 {
  const dims = [0, 1, 2].map((d) => {
    const vals = nodes.map((n) => n.coords5d[d] ?? 0);
    const lo = Math.min(...vals, 0);
    const hi = Math.max(...vals, 0);
    return { lo, span: hi - lo || 1 };
  });
  const R = 6;
  return (c: number[]) => {
    const p = pos3(c);
    return new THREE.Vector3(
      ...([0, 1, 2].map((d) => {
        const dim = dims[d]!;
        return (((p[d] ?? 0) - dim.lo) / dim.span) * 2 * R - R;
      }) as [number, number, number]),
    );
  };
}

export function StyleSpaceScene({
  nodes,
  teammates,
  showTeammates,
  selected,
  onSelect,
  onHover,
  unknownVector,
  onInitError,
}: Props) {
  const mountRef = useRef<HTMLDivElement | null>(null);
  // Stash the latest callbacks/props so the (once) init effect reads fresh values without re-init.
  const cb = useRef({ onSelect, onHover, showTeammates, selected });
  cb.current = { onSelect, onHover, showTeammates, selected };

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    let raf = 0;
    let renderer: THREE.WebGLRenderer | null = null;
    let disposed = false;

    try {
      const width = mount.clientWidth || 640;
      const height = mount.clientHeight || 480;
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
      camera.position.set(0, 0, 20);
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setSize(width, height);
      mount.appendChild(renderer.domElement);

      scene.add(new THREE.AmbientLight(0xffffff, 0.7));
      const point = new THREE.PointLight(0xffffff, 0.8);
      point.position.set(10, 10, 20);
      scene.add(point);

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.autoRotate = true;
      controls.autoRotateSpeed = CAMERA_ROTATE_DEG_PER_SEC * (60 / 360); // three units ≈ deg/frame@60fps

      const scale = makeScaler(nodes);
      const meshByCode = new Map<string, THREE.Mesh>();
      const geo = new THREE.SphereGeometry(0.35, 24, 24);
      for (const n of nodes) {
        const colour = new THREE.Color(teamColour(n.team));
        const mat = new THREE.MeshStandardMaterial({ color: colour, emissive: colour, emissiveIntensity: 0.4 });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.copy(scale(n.coords5d));
        mesh.userData.code = n.code;
        scene.add(mesh);
        meshByCode.set(n.code, mesh);
      }

      // Teammate lines.
      const lineGroup = new THREE.Group();
      lineGroup.visible = cb.current.showTeammates;
      for (const t of teammates) {
        const a = meshByCode.get(t.a);
        const b = meshByCode.get(t.b);
        if (!a || !b) continue;
        const g = new THREE.BufferGeometry().setFromPoints([a.position, b.position]);
        const line = new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x2b2b2b }));
        lineGroup.add(line);
      }
      scene.add(lineGroup);

      // Unknown-driver white sphere: drift from origin to its (scaled) 5D position over unknownDrift.
      let unknownMesh: THREE.Mesh | null = null;
      let unknownTarget: THREE.Vector3 | null = null;
      let unknownStart = 0;
      if (unknownVector && unknownVector.length > 0) {
        unknownMesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.6 }));
        unknownMesh.position.set(0, 0, 0);
        unknownTarget = scale(unknownVector);
        scene.add(unknownMesh);
      }

      // Hover + click via raycaster.
      const raycaster = new THREE.Raycaster();
      const pointer = new THREE.Vector2();
      const pick = (ev: MouseEvent): THREE.Mesh | null => {
        const rect = renderer!.domElement.getBoundingClientRect();
        pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
        pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(pointer, camera);
        const hit = raycaster.intersectObjects([...meshByCode.values()])[0];
        return (hit?.object as THREE.Mesh) ?? null;
      };
      const onMove = (ev: PointerEvent) => cb.current.onHover((pick(ev)?.userData.code as string) ?? null);
      const onClick = (ev: MouseEvent) => {
        const code = pick(ev)?.userData.code as string | undefined;
        if (code) cb.current.onSelect(code);
      };
      renderer.domElement.addEventListener('pointermove', onMove);
      renderer.domElement.addEventListener('click', onClick);

      const onResize = () => {
        const w = mount.clientWidth || width;
        const h = mount.clientHeight || height;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer!.setSize(w, h);
      };
      window.addEventListener('resize', onResize);

      const clock = new THREE.Clock();
      const loop = () => {
        if (disposed) return;
        raf = requestAnimationFrame(loop);
        lineGroup.visible = cb.current.showTeammates;
        if (unknownMesh && unknownTarget) {
          unknownStart += clock.getDelta();
          const t = Math.min(1, (unknownStart * 1000) / DURATION.unknownDrift);
          unknownMesh.position.lerpVectors(new THREE.Vector3(0, 0, 0), unknownTarget, t);
        }
        controls.update();
        renderer!.render(scene, camera);
      };
      loop();

      return () => {
        disposed = true;
        cancelAnimationFrame(raf);
        window.removeEventListener('resize', onResize);
        renderer?.domElement.removeEventListener('pointermove', onMove);
        renderer?.domElement.removeEventListener('click', onClick);
        controls.dispose();
        geo.dispose();
        renderer?.dispose();
        if (renderer?.domElement.parentNode === mount) mount.removeChild(renderer.domElement);
      };
    } catch {
      // WebGL init failed → silent fallback to the D3 2D view (master §10.6). Never throws to React.
      if (!disposed) onInitError?.();
      return () => {
        disposed = true;
      };
    }
    // Re-init only when the data or the unknown projection changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodes, teammates, unknownVector]);

  return <div ref={mountRef} data-testid="style-space-scene" className="h-[480px] w-full" />;
}
