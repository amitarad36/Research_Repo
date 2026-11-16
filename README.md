# Research proposal and backbone

Two documents serve different purposes:

| Document | PDF | Editable source |
|---|---|---|
| **Proposal** — concise research questions, motivation and plan, based on the supplied ZIP | [research_proposal.pdf](research_proposal.pdf) | [research_proposal_main.tex](research_proposal_main.tex) |
| **Research backbone** — detailed derivations, checked experiments and limitations | [backbone/research_backbone.pdf](backbone/research_backbone.pdf) | [backbone/research_backbone.tex](backbone/research_backbone.tex) |

The proposal source keeps its original filename so the existing editor continues to watch it. Its compiled PDF is deliberately named **research_proposal.pdf**. The latest detailed local source, including cycles 33–34, is the editable backbone. The ZIP's backbone PDF is an earlier snapshot; it is preserved in the import archive.

## Folder map

- `backbone/`: current long technical document and its PDF.
- `research/`: persistent state, research log, experiments, scripts, data and evidence. Existing internal paths are preserved.
- `presentation/`: existing slides, figures and speaker notes.
- `archive/imports/2026-11-16/`: original ZIP, supplied README and supplied backbone PDF.
- `archive/2026-11-16-before-organization/`: original local TeX and PDF before the split.
- `build/`: reproducible compiler files, preview PDFs and pass logs for both documents.
- `build/legacy/`: earlier root-level compiler files retained during organization.
- `.vscode/`: project-local LaTeX Workshop recipes.

## Build

From this folder in PowerShell:

```powershell
.\build_documents.ps1
.\build_documents.ps1 -Document proposal
.\build_documents.ps1 -Document backbone
```

The script runs pdfLaTeX twice, puts intermediate files under `build/`, and refreshes the named PDFs above. It uses an installed `pdflatex` on PATH and disables shell escape. In VS Code, use the **Build current research document** recipe; **Build both research documents** refreshes both PDFs. The automatic recipe selects the backbone when its source is active, otherwise the proposal.

## Editing policy

Keep the proposal concise and forward-looking. Put new derivations, exploratory results, numerical audits, and failures in the backbone and `research/`. Revise the proposal only when findings materially change its questions, assumptions, scope, plan or defensible claims. Do not append the research diary to the proposal. Claims and references still require scholarly review before submission; successful compilation is not validation of novelty or accuracy.

The recurring research task uses these two roles. Original user-supplied content and prior source versions are preserved in the archive. Nothing under the synced ChatGPT project's `sources/` was changed.

Organization record: [research/notes/2026-11-16-document-organization/ORGANIZATION.md](research/notes/2026-11-16-document-organization/ORGANIZATION.md).
