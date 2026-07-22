import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

// Render next/link as a plain anchor (no App Router context needed in a unit test).
vi.mock('next/link', () => ({
  default: ({ children, href, ...rest }: { children: React.ReactNode; href: string }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
}));

import { DriverCard, type Metric } from './DriverCard';

const OK: Metric = { value: 0.8, loading: false, error: false };
const ERRORED: Metric = { value: null, loading: false, error: true };
const driver = {
  id: 'VER',
  code: 'VER',
  team: 'Red Bull',
  radar_values: { throttle: 0.9, braking: 0.3, speed: 0.7, gear: 0.4, engine: 0.6, cornering: 0.5 },
};

describe('DriverCard (master §10.2)', () => {
  it('links to the profile and paints the team-colour bar', () => {
    render(<DriverCard driver={driver} compatibility={OK} stability={OK} animate={false} />);
    expect(screen.getByTestId('driver-card-VER')).toHaveAttribute('href', '/drivers/VER');
    expect(document.querySelector('rect[fill="#3671C6"]')).not.toBeNull(); // Red Bull colour
  });

  it('applies the LOW_CONFIDENCE treatment: dashed border + warning badge + dimmed radar', () => {
    render(
      <DriverCard
        driver={driver}
        compatibility={OK}
        stability={OK}
        confidenceState="LOW_CONFIDENCE"
        animate={false}
      />,
    );
    expect(screen.getByTestId('confidence-badge')).toHaveTextContent('LOW CONFIDENCE');
    expect(screen.getByTestId('fingerprint-radar')).toHaveClass('opacity-40');
    expect(document.querySelector('.border-dashed')).not.toBeNull();
  });

  it('renders "—" for an errored/null metric — never a fabricated number (div #47)', () => {
    render(<DriverCard driver={driver} compatibility={ERRORED} stability={OK} animate={false} />);
    expect(screen.getByTestId('compat')).toHaveTextContent('—');
    expect(screen.getByTestId('stability')).toHaveTextContent('80%');
  });

  it('reveals the top-3 SHAP peek on hover', () => {
    render(<DriverCard driver={driver} compatibility={OK} stability={OK} animate={false} />);
    expect(screen.queryByTestId('hover-shap')).toBeNull();
    fireEvent.mouseEnter(screen.getByTestId('driver-card-VER'));
    expect(screen.getByTestId('hover-shap')).toBeInTheDocument();
  });
});
