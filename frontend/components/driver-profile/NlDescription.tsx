// Auto-generated natural-language style description (master §6/§10.3). The generic fallback is the
// master's verbatim "no dominant domain" sentence. Fades on change via a CSS opacity transition.
const GENERIC =
  'This driver shows balanced characteristics across all domains with no single dominant style trait.';

export function NlDescription({ text }: { text: string }) {
  return (
    <p
      data-testid="nl"
      className="text-sm italic leading-relaxed text-text-secondary transition-opacity duration-300"
    >
      {text && text.trim().length > 0 ? text : GENERIC}
    </p>
  );
}
