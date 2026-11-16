# Repository migration — 16 November 2026

- Canonical folder: `C:\Users\amita\source\repos\Research_Repo`.
- Remote: https://github.com/amitarad36/Research_Repo (public), branch `main`.
- Previous path: `C:\Users\amita\source\repos\Detach Proposal Presentation`.
- The previous path is a junction to the canonical folder, not duplicate data.
- All 536 original files matched the pre-move SHA-256 inventory immediately after rename.
- Original source/evidence snapshot: commit `4b859fd6d2ddb188db65a4b902ac29b580c44c15`.
- `.gitignore` excludes generated `build/` output and Python bytecode/cache files only.
- `.gitattributes` preserves exact file bytes without line-ending normalization.
- Original TeX content and archived scripts are unchanged. Historical hardcoded paths
  continue to resolve locally through the junction. A fresh clone elsewhere does
  not automatically provide those paths; the document build itself is portable.
- The active research automation now points to Research_Repo; its schedule remains
  twice daily at 09:00 and 17:00. No automatic push behavior was added.
- No synced ChatGPT project sources were changed.

`before_manifest.json` records the pre-move inventory. Final build and Git upload
checks are recorded in `validation.json` after completion.
