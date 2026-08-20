// Centralised client-side error log (F16 spec §6 "client log" column). One choke point so it can
// later be swapped for a telemetry sink. Not user-facing — the user sees the typed UI state; this
// makes the failure observable to an operator, so it is never silent even when handled gracefully.
export function logClientError(code: string, detail: string, context?: string): void {
  const prefix = context ? `${context}: ` : '';
  // eslint-disable-next-line no-console
  console.error(`[apex] ${prefix}${code}${detail ? `: ${detail}` : ''}`);
}
