import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import type { DriverSummary } from '@/lib/api/types';

import { ProfileHeader } from './ProfileHeader';

describe('ProfileHeader', () => {
  it('renders race number, full name, team and code tag', () => {
    render(
      <ProfileHeader
        driver={
          {
            id: 'VER',
            code: 'VER',
            team: 'Red Bull Racing',
            full_name: 'Max Verstappen',
            radar_values: {},
          } as DriverSummary
        }
      />,
    );
    const header = screen.getByTestId('profile-header');
    expect(header).toHaveTextContent('Max Verstappen'); // caps come from CSS, not the DOM text
    expect(header).toHaveTextContent('Red Bull Racing');
    expect(header).toHaveTextContent('#VER');
    expect(header).toHaveTextContent('1'); // race number
  });

  it('falls back to the code when identity data is absent — nothing invented', () => {
    render(
      <ProfileHeader
        driver={{ id: 'ZZZ', code: 'ZZZ', team: null, full_name: null, radar_values: {} } as DriverSummary}
      />,
    );
    const header = screen.getByTestId('profile-header');
    expect(header).toHaveTextContent('ZZZ');
    expect(header).toHaveTextContent('Team unknown');
  });
});
