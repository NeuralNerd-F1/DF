import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { ProfileRadar } from './ProfileRadar';

const values = { throttle: 0.9, braking: 0.3, speed: 0.7, gear: 0.4, engine: 0.6, cornering: 0.5 };
const fieldMean = { throttle: 0.5, braking: 0.3, speed: 0.5, gear: 0.5, engine: 0.5, cornering: 0.5 };

describe('ProfileRadar tooltip (master §10.3)', () => {
  it('shows domain, score, and field-mean comparison on axis hover', () => {
    render(<ProfileRadar values={values} fieldMean={fieldMean} animate={false} />);
    expect(screen.queryByTestId('radar-tooltip')).toBeNull();

    const axis = document.querySelector('text[data-domain="throttle"]');
    expect(axis).not.toBeNull();
    fireEvent.mouseEnter(axis as Element);

    const tip = screen.getByTestId('radar-tooltip');
    expect(tip).toHaveTextContent('throttle');
    expect(tip).toHaveTextContent('0.90'); // the driver's score
    expect(tip).toHaveTextContent('field mean 0.50');
    expect(tip).toHaveTextContent('+0.40'); // above the field
  });
});
