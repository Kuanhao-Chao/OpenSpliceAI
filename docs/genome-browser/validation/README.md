# Browser verification evidence

Run `npm run audit -- --output=validation/chromium --project=chromium` and
the corresponding Firefox/WebKit commands to keep engine-specific screenshots
and failure traces. The Ubuntu documentation workflow runs all engines and
uploads the artifacts. Build and unit-test logs can be captured here.

`results.json` records the tested source/data identities, actual check results,
bundle sizes and outstanding publication dependencies. Large transient traces
and screenshots are not committed to the software repository.
