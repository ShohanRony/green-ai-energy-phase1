# Run every test file in this directory. One-line usage:
#   Rscript analysis/tests/run_all.R

test_dir <- dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE),
                                             value = TRUE)))
test_files <- sort(list.files(test_dir, pattern = "^test_.*\\.R$", full.names = TRUE))

total_failed <- 0
for (f in test_files) {
  cat("--- ", basename(f), " ---\n", sep = "")
  status <- tryCatch({
    system2("Rscript", args = f, stdout = "", stderr = "")
  }, error = function(e) 1)
  if (!identical(status, 0L) && !identical(status, 0)) {
    total_failed <- total_failed + 1
    cat("FAILED: ", basename(f), "\n", sep = "")
  }
}

if (total_failed > 0) {
  cat("\n", total_failed, " of ", length(test_files), " test file(s) failed\n", sep = "")
  quit(status = 1)
} else {
  cat("\nAll ", length(test_files), " test file(s) passed\n", sep = "")
}
