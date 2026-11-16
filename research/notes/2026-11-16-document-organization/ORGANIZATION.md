# Proposal and backbone organization
16 November 2026. User explicitly requested using research_proposal_package.zip, splitting the previous detailed work into a backbone, and organizing/renaming the local folder.

- Root research_proposal_main.tex now holds the supplied concise proposal source. Existing source editor path is preserved.
- Public short PDF is research_proposal.pdf.
- backbone/research_backbone.tex holds the latest detailed local source, with only identifying comments/subtitle changed. It includes cycles33 and34.
- Public detailed PDF is backbone/research_backbone.pdf, rebuilt from that source.
- ZIP and supplied README/backbone PDF preserved under archive/imports/2026-11-16. The supplied long PDF had24pages and includes cycle33 but not the latest cycle34 paragraph; it is a snapshot rather than the editable authority.
- Prior local source/PDF and state/log preserved under archive/2026-11-16-before-organization. Existing localPDF also had24pages and predated the latest source.
- Existing research internals and presentation files retained in place to preserve code/data paths.
- Known reproducible root-level compiler files moved individually under build/legacy; no research material deleted.
- build_documents.ps1 compiles both documents twice with no shell escape, puts intermediate/preview files under build and copies deliverable PDFs to their named locations. VSCode recipes use this script.
- Research state/log and recurring instructions distinguish short proposal from long backbone.
- This organization does not claim the supplied proposal's citations or conjectures have been newly validated. Scientific content was not rewritten during import.

Hashes, compiler records, PDF page counts and visual QA are retained alongside this note.

Completed: both built-in compiler checks succeeded. Native two-pass builds produced 12-page proposal and 24-page backbone, with no overfull boxes, undefined references or compiler errors. All pages visually reviewed; cover indentation and an orphan heading were fixed in the proposal, and long-line wrapping improved in the backbone. Scientific text unchanged. Exact Windows PowerShell command used by the VS Code recipe passed. Automation updated and verified ACTIVE at the existing 09:00/17:00 schedule. Source/PDF/build hashes recorded in organization_manifest.json.
