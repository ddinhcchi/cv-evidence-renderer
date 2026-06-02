# `docs/` — landing page

`docs/index.html` is the static landing page served by GitHub Pages.

## Deploy

In the GitHub repo settings: **Settings → Pages**, set:

- **Source**: Deploy from a branch
- **Branch**: `main`
- **Folder**: `/docs`

After a few seconds the page is live at
`https://ddinhcchi.github.io/cv-evidence-renderer/`.

## Edit

Single file, embedded CSS, no build step. Open `index.html` directly in
a browser to preview locally:

```bash
open docs/index.html
```

The demo GIF is referenced via an absolute `raw.githubusercontent.com`
URL so we don't have to keep a second copy of the binary inside `docs/`.
