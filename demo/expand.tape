# tok-adapt expand demo — VHS tape script
#
# Not run in this environment: no `vhs` binary is installed here, and this
# environment has no real TTY to record from. To generate the GIF yourself:
#
#   1. Install VHS: https://github.com/charmbracelet/vhs (`brew install vhs`,
#      or `go install github.com/charmbracelet/vhs@latest`)
#   2. From the repo root: vhs demo/expand.tape
#   3. This produces demo/expand.gif
#
# Uses the tiny public checkpoint (hf-internal-testing/tiny-random-gpt2).
# Measured locally (warm HF cache): ~11s end-to-end, mostly torch/transformers
# import overhead rather than tok-adapt's own logic (sub-tokenizer training +
# merging complete in under 1s). Budget more on a cold cache (first-time
# model download) or a slower machine. Swap --model for a real checkpoint if
# you want to record a production run instead.

Output demo/expand.gif

Set Shell "bash"
Set FontSize 16
Set Width 1200
Set Height 600
Set Theme "Dracula"
Set TypingSpeed 40ms
Set Padding 20

Type "tok-adapt expand \"
Enter
Sleep 300ms
Type "  --model hf-internal-testing/tiny-random-gpt2 \"
Enter
Sleep 300ms
Type "  --corpus examples/domain_corpus.txt \"
Enter
Sleep 300ms
Type "  --add-vocab-size 2000 \"
Enter
Sleep 300ms
Type "  --output ./expanded-model"
Sleep 500ms
Enter

Sleep 15s
