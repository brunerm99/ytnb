# ytnb

Supplemental [marimo](https://marimo.io) notebooks for my RF/radar videos,
published on Cloudflare Pages and embedded in
[marshallbruner.com/resources](https://marshallbruner.com/resources). Every
notebook is exported to WebAssembly (Pyodide), so it runs entirely in the
reader's browser — no server, no install.

Each notebook is published at `/<slug>/`; there is no index page at the site
root.

```html
<iframe
  src="https://notebooks.marshallbruner.com/gain/"
  style="width: 100%; height: 90vh; border: 0"
  loading="lazy"
  title="Gain — how to read an RF amplifier datasheet"
></iframe>
```

## Layout

```
gain.py, reflectivity.py   notebooks (one per page on the site)
public/                    data files (.s2p, .csv, ...) served at /public/
site/_headers              Cloudflare Pages headers (CORS, caching)
build.sh                   exports every notebook into dist/
.github/workflows/         builds and deploys on push to main
```

## Local preview

```sh
./build.sh
python3 -m http.server --directory dist
```

Then open the printed URL. To work on a notebook the normal way:

```sh
uv run marimo edit --sandbox gain.py
```

`--sandbox` installs the packages listed in the notebook's inline script
metadata (the `# /// script` block at the top), which is the same list the
browser installs at runtime. **Keep that block accurate — it is what makes the
WASM build work.**

## Adding a notebook

1. Drop `<slug>.py` in the repo root, with a `# /// script` metadata block
   listing its dependencies.
2. Add `<slug>` to `NOTEBOOKS` in `build.sh`.
3. Push. CI builds and deploys it at `/<slug>/`.

Packages must be pure Python or shipped with Pyodide (numpy, scipy, pandas,
matplotlib, scikit-rf, sympy, ... all work; anything needing a compiled
extension not in Pyodide, hardware access, or threads will not).

## Data files

Anything in `public/` is published twice:

- next to each notebook (`/<slug>/public/<file>`), which is what
  `mo.notebook_location() / "public" / "<file>"` resolves to, and
- at the site root (`/public/<file>`) with `Access-Control-Allow-Origin: *`,
  so any other notebook or site can fetch it.

Pattern for reading one in a way that works both locally and in the browser
(see `gain.py`):

```python
s2p_source = str(mo.notebook_location() / "public" / "ADL8142ACPZN_25degC.s2p")
if s2p_source.startswith("http"):
    s2p_path = pathlib.Path("ADL8142ACPZN_25degC.s2p")
    s2p_path.write_bytes(urllib.request.urlopen(s2p_source).read())
else:
    s2p_path = pathlib.Path(s2p_source)
amp = rf.Network(str(s2p_path))
```

`skrf` wants a real file path, hence the download-to-disk step; in the browser
that "disk" is Pyodide's in-memory filesystem. Cloudflare's free tier allows
25 MiB per file and 20,000 files per deployment, so touchstone files are free
to host in bulk.

## Cloudflare setup (one time)

1. Create the Pages project and do the first deploy by hand:

   ```sh
   npx wrangler login
   ./build.sh
   npx wrangler pages project create ytnb --production-branch main
   npx wrangler pages deploy dist --project-name ytnb
   ```

2. Create an API token at
   <https://dash.cloudflare.com/profile/api-tokens> using the
   **Cloudflare Workers** template, or a custom token with
   `Account → Cloudflare Pages → Edit`.

3. Add the two repo secrets:

   ```sh
   gh secret set CLOUDFLARE_API_TOKEN --repo brunerm99/ytnb
   gh secret set CLOUDFLARE_ACCOUNT_ID --repo brunerm99/ytnb
   ```

4. Point `notebooks.marshallbruner.com` at it. `marshallbruner.com` uses
   Porkbun's nameservers, not Cloudflare's, so this is a plain CNAME:

   - Cloudflare dashboard → Workers & Pages → `ytnb` → Custom domains → Set up
     a custom domain → `notebooks.marshallbruner.com`.
   - Porkbun DNS: add `CNAME  notebooks  ytnb.pages.dev`. (There is a wildcard
     record on the zone today pointing at Porkbun parking; the specific record
     wins.)

   Cloudflare issues the certificate once the CNAME resolves, usually within a
   few minutes.

After that, every push to `main` rebuilds and redeploys. The site is free:
unlimited requests and bandwidth, 500 Pages builds/month (this repo doesn't use
Cloudflare's builders at all — GitHub Actions does the build and uploads the
result).

## What the reader downloads

First visit pulls Pyodide plus numpy, scipy, pandas, matplotlib and scikit-rf —
roughly 40–50 MB, about a minute before the plots appear. It is cached
afterwards. `skrf` is what drags in scipy and pandas; if that ever feels too
heavy, `gain.py` only uses it to parse a touchstone file, which numpy can do in
about fifteen lines.
