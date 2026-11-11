/// <reference types="node" />
import adapter from '@sveltejs/adapter-static';
import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vitest/config';

// BASE_PATH: '' for a custom domain / root deploy, '/repo-name' for GitHub project pages.
const base = process.env.BASE_PATH ?? '';

export default defineConfig({
	plugins: [
		sveltekit({
			adapter: adapter({ pages: 'build', assets: 'build', strict: true }),
			paths: { base: base as '' | `/${string}`, relative: true },
			prerender: { entries: ['*'], handleHttpError: 'fail' }
		})
	],
	test: {
		include: ['src/**/*.test.ts'],
		environment: 'node'
	}
});
