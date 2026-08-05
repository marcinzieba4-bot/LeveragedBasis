# Logic page — deployment notes

Self-contained static page: `index.html` + `assets/*.png`. No build step,
no external dependencies (fonts/JS/CSS are all inline or system fonts), so
it can be dropped as-is into the existing website's static hosting.

## Deploying (S3 + CloudFront, matching the other project)

1. Sync this folder to the path the site's **Logic** tab points at, e.g.:
   ```
   aws s3 sync web/logic/ s3://<bucket>/logic/ --delete
   ```
2. Invalidate CloudFront so the new page is served immediately:
   ```
   aws cloudfront create-invalidation --distribution-id <DIST_ID> --paths "/logic/*"
   ```
3. The site's own nav/shell is untouched — only files under `/logic/` are
   added. Point the existing "Logic" tab's link/route at `/logic/index.html`
   (or `/logic/` if the bucket has `index.html` as the default root object
   for that prefix).

## Live PnL data

`index.html` fetches `data/live_pnl.json` (relative to itself) for the Live
Performance section. That file does not exist yet — `fetch_derive_pnl.py`
(repo root) generates it from the Derive account on a schedule and it should
be uploaded alongside `index.html` on the same sync. Until it exists, the
page shows "Live feed not yet connected" instead of fabricated numbers.

## Regenerating the backtest charts

If `backtest.py` is re-run with fresher data, re-copy the three PNGs into
`assets/` and update the numbers in `index.html` (KPI strip, KPI grid,
quarterly table, and the descriptive prose) to match.
