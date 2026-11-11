export type Theme = 'light' | 'dark';

/** Reactive theme state. The initial value is applied pre-paint by the inline script in app.html. */
class ThemeState {
	current = $state<Theme>('light');

	init(): void {
		const t = document.documentElement.dataset.theme;
		this.current = t === 'dark' ? 'dark' : 'light';
	}

	set(t: Theme): void {
		this.current = t;
		document.documentElement.dataset.theme = t;
		try {
			localStorage.setItem('theme', t);
		} catch {
			/* storage unavailable: theme still applies for this page view */
		}
	}

	toggle(): void {
		this.set(this.current === 'dark' ? 'light' : 'dark');
	}
}

export const theme = new ThemeState();
