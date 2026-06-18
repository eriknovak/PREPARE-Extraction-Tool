# Monitor Dashboard UI Restyle — Implementation Plan

> **For agentic workers:** execute task-by-task. Steps use checkbox (`- [ ]`) syntax. This is a styling/consistency refactor — behavior must not change. Verify each task with `npm run build` + `npm run lint`; visually confirm in Storybook/the running app.

**Goal:** Restyle the ported Monitor dashboard so it matches the rest of the frontend — CSS Modules + BEM, reused shared components, design tokens — with **no behavioral change** to data flow or endpoints.

**Architecture:** Replace the page's 28 inline-style blocks with a `Monitor/styles.module.css` (BEM), swap raw HTML elements for existing components (`Button`, `StatCard`, `Select`, `Toast`, `Dropdown`), introduce a reusable `Card` panel + a `ProgressBar` + a `LabelToggle`, and align all colors to design tokens (incl. recharts + the SVG heatmap). Surface the new `exact_f1`/`relaxed_f1` metrics the backend already returns.

**Tech Stack:** React 19, TypeScript, Vite, CSS Modules (BEM), `classnames`, recharts v3.8 (already installed).

**Reference inventory:** design tokens & components mapped in this session (Button, StatCard, Layout, Header, Dropdown, Select, Toast/ToastContainer, ProcessingBadge, LoadingSpinner; recharts already used).

**Scope:** `frontend/src/pages/Monitor/*`, plus small additions under `frontend/src/components/` and `frontend/src/index.css`. No backend/API changes (response shapes already include `exact_f1`/`relaxed_f1`).

---

## Design decisions (baked in; flag if you disagree)
1. **Formalize design tokens** as CSS custom properties in `src/index.css` `:root` (colors/spacing/radius), reusing the values already recurring across modules (primary `#144d70`, success `#269b6f`, error `#dc2626`, border `#e5e7eb`, text `#374151`/`#6b7280`, radii 4/6/8/12). All Monitor styles reference tokens; no hardcoded hexes.
2. **Add a reusable `Card` component** for dashboard panels (the white-bg/rounded/shadow pattern already used ad-hoc on other pages) instead of the inline `SectionCard`.
3. **Per-label evaluation chart shows exact vs relaxed F1** (grouped bars) since the backend now returns both, plus precision/recall — replaces the single-F1 fallback.

---

## File Structure
- Create: `frontend/src/components/Card/{index.tsx,styles.module.css}` — generic panel (title + children).
- Create: `frontend/src/components/ProgressBar/{index.tsx,styles.module.css}` — labeled progress bar (if no existing one; inventory found none formal).
- Create: `frontend/src/pages/Monitor/styles.module.css` — page BEM styles.
- Create: `frontend/src/pages/Monitor/LabelSelector` styles (fold into page module or its own module).
- Modify: `frontend/src/pages/Monitor/index.tsx` — remove inline styles, use components + module classes.
- Modify: `frontend/src/pages/Monitor/LabelSelector.tsx` — use `Button`/toggle + module classes.
- Modify: `frontend/src/index.css` — add `:root` tokens.
- Optional: `frontend/src/components/Card/index.stories.tsx`, `ProgressBar/index.stories.tsx` (project uses Storybook).

---

## Phase 1 — Foundations

### Task 1: Design tokens in `:root`
**Files:** Modify `frontend/src/index.css`
- [ ] Add a `:root` block with custom properties for the recurring values (do not change existing global rules, just add tokens):
  ```css
  :root {
    --color-primary: #144d70;
    --color-primary-dark: #11415e;
    --color-success: #269b6f;
    --color-error: #dc2626;
    --color-info: #3b82f6;
    --color-warning: #f59e0b;
    --color-text: #374151;
    --color-text-muted: #6b7280;
    --color-border: #e5e7eb;
    --color-surface: #ffffff;
    --color-surface-alt: #f9fafb;
    --space-1: 8px; --space-2: 12px; --space-3: 16px; --space-4: 24px; --space-5: 32px;
    --radius-sm: 4px; --radius-md: 6px; --radius-lg: 8px; --radius-xl: 12px;
    --shadow-card: 0 1px 3px rgba(0,0,0,0.08);
  }
  ```
- [ ] `npm run build` → success. Commit: `style(frontend): add design tokens to :root`.

### Task 2: `Card` panel component
**Files:** Create `frontend/src/components/Card/index.tsx` + `styles.module.css`
- [ ] `index.tsx` (default export) props: `title?: string`, `actions?: React.ReactNode`, `className?: string`, `children`. Render `.card` with optional `.card__header` (title + actions) and `.card__body`. Use `classnames`.
- [ ] `styles.module.css`: `.card` (surface bg, `--radius-xl`, `--shadow-card`, `--space-4` padding, border `--color-border`), `.card__header` (flex, space-between, title `--color-primary-dark`), `.card__body`.
- [ ] (Optional) `index.stories.tsx`.
- [ ] `npm run build` + `npm run lint` (Card only) → clean. Commit: `feat(frontend): add reusable Card panel component`.

### Task 3: `ProgressBar` component
**Files:** Create `frontend/src/components/ProgressBar/index.tsx` + `styles.module.css`
- [ ] Props: `value: number` (0–100), `label?: string`. Render `.progress` track + `.progress__fill` (width = value%), optional `.progress__label`. Fill color `--color-success`.
- [ ] Build + lint clean. Commit: `feat(frontend): add ProgressBar component`.

---

## Phase 2 — Monitor page restyle (no behavior change)

### Task 4: Page shell + CSS module + tokens
**Files:** Create `frontend/src/pages/Monitor/styles.module.css`; Modify `Monitor/index.tsx`
- [ ] Add `.monitor`, `.monitor__title`, `.monitor__grid` (`grid-template-columns: 1fr 1fr; gap: var(--space-4)`), `.monitor__row`, `.monitor__section` classes.
- [ ] Replace the page heading inline style (`<h1 style=...>`) with `<h1 className={styles.monitor__title}>`. Keep `usePageTitle` if present (add if missing, matching other pages).
- [ ] Replace the two-column `<div style={{display:'grid',...}}>` with `className={styles.monitor__grid}`.
- [ ] Build → success (page still renders). Commit: `style(monitor): page shell to CSS module`.

### Task 5: Panels → `Card`
**Files:** Modify `Monitor/index.tsx`
- [ ] Replace the inline `SectionCard` wrapper and each panel container (dataset selector, stats, label selector, training config, training progress, evaluation, heatmap) with the `Card` component (`title` prop for headings). Remove the inline `SectionCard` definition.
- [ ] Build → success. Commit: `style(monitor): use Card for all panels`.

### Task 6: Dataset selector + stats
**Files:** Modify `Monitor/index.tsx`
- [ ] Dataset selector buttons: already use `Button` — move any inline spacing to `.monitor__dataset-list` (flex wrap, gap `--space-2`). Active dataset uses `Button variant="primary"`, others `variant="outline"` (already the case) — just remove inline styles.
- [ ] Stats: keep `StatCard` (Records/Terms); wrap in `.monitor__stats` (flex, gap). Remove inline row style.
- [ ] Build → success. Commit: `style(monitor): dataset selector and stats`.

### Task 7: LabelSelector → tokenized toggles
**Files:** Modify `Monitor/LabelSelector.tsx`; add classes to `Monitor/styles.module.css` (or a `LabelSelector` module)
- [ ] Replace the inline-styled `<div>` toggles with buttons styled via module classes: `.label-toggle`, `.label-toggle--active` (active bg `--color-success`, text white; inactive surface + border). Keep the same onClick/selected behavior and the `(count)` label.
- [ ] Use `classnames` for the active modifier. No hardcoded `#4caf50`.
- [ ] Build → success. Commit: `style(monitor): tokenized label toggles`.

### Task 8: Training config form → Select + form classes
**Files:** Modify `Monitor/index.tsx`
- [ ] Replace raw `<select>` (train/eval split) and the run dropdown with the existing `Select` component (match its props). Replace the model radio group + custom-model `<input type="text">` with module-classed form rows (`.monitor__field`, `.monitor__label`, `.monitor__input`) using token colors/focus. Keep Start/Stop as `Button` (primary / danger). Keep the training-status text but class it `.monitor__status--{running|error|idle}`.
- [ ] Verify the form still produces the same `start` payload `{dataset_id, labels, base_model, val_ratio}`. Build → success. Commit: `style(monitor): training config form uses Select + form classes`.

### Task 9: Progress + charts tokenized
**Files:** Modify `Monitor/index.tsx`
- [ ] Replace the raw progress `<div>` bars with the `ProgressBar` component.
- [ ] LineChart (loss): set `Line stroke="var(--color-error)"`-equivalent — recharts needs a concrete color, so import token hexes from a small `chartColors` const (e.g. `export const CHART = { loss: '#dc2626', precision: '#144d70', recall: '#3b82f6', exactF1: '#269b6f', relaxedF1: '#f59e0b' }`) in `Monitor/chartColors.ts`, sourced from the same palette as the tokens. Add `Tooltip`/`Legend` for consistency.
- [ ] BarChart (per-label evaluation): render grouped bars for **exact_f1, relaxed_f1, precision, recall** from `per_label[label]` (data now provided by backend). Replace the `readF1` single-bar logic. Build → success. Commit: `style(monitor): tokenized charts with exact/relaxed F1`.

### Task 10: Heatmap colors + tooltip
**Files:** Modify `Monitor/index.tsx`
- [ ] Keep the custom SVG heatmap structure but route its `getColor()` gradient and the tooltip box through the `chartColors`/token palette (no `#111`/inline ad-hoc hexes; tooltip uses `--color-primary-dark` bg). Move the metric selector to the `Select` component. Class the SVG container `.monitor__heatmap`.
- [ ] Build → success. Commit: `style(monitor): heatmap colors via palette`.

### Task 11: Alerts → Toast
**Files:** Modify `Monitor/index.tsx`
- [ ] Replace the fixed-position inline alert `<div>` with the existing `Toast`/`ToastContainer` component (match its API). Remove the inline fixed-position styling.
- [ ] Build → success. Commit: `style(monitor): use Toast for alerts`.

---

## Phase 3 — Verify

### Task 12: Full verification
- [ ] `cd frontend && npm run build` → success (no TS errors).
- [ ] `npm run lint` → no NEW errors in touched files (pre-existing errors elsewhere out of scope).
- [ ] Grep the Monitor page for residual inline styles: `grep -n "style={{" src/pages/Monitor/*.tsx` → **zero** matches.
- [ ] Grep for stray hex colors in Monitor TSX: `grep -nE "#[0-9a-fA-F]{3,6}" src/pages/Monitor/*.tsx` → only the documented `chartColors` import, no inline hexes.
- [ ] Run the app against the live stack; confirm the dashboard renders, dataset switch works, training start/stop works, charts + heatmap display, alerts show via Toast. (Behavior identical to pre-restyle.)
- [ ] Commit any fixups: `style(monitor): restyle verification fixups`.

---

## Out of scope (separate follow-ups)
- WebSocket live-update message shapes (`training_start`, `train_log`) verification against backend.
- Model-settings UI / `UserModelPreference` (deferred bucket C).
- Any new dashboard features beyond restyling existing panels.

## Self-review notes
- **Behavior preserved:** every task is visual/structural; API calls and payloads untouched (Tasks 8/9 explicitly re-verify the start payload and data shapes).
- **Consistency:** all panels via `Card`, all inputs via `Select`/form classes, all colors via tokens/`chartColors`, alerts via `Toast` — matching existing pages (Datasets, VocabularyUpload).
- **New data surfaced:** Task 9 renders `exact_f1`/`relaxed_f1` the backend already returns.
