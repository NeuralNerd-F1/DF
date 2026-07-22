import { useEffect, useLayoutEffect } from 'react';

// useLayoutEffect on the client — runs the chart's entrance-collapse BEFORE paint, so there is no
// one-frame flash of the finished radar (code-review #2). useEffect on the server avoids React's
// "useLayoutEffect does nothing on the server" warning (effects never run during SSR anyway).
export const useIsomorphicLayoutEffect =
  typeof window !== 'undefined' ? useLayoutEffect : useEffect;
