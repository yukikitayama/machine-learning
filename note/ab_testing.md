## A/B Testing

- Test incremental changes (UX changes, new features, ranking and page load times) to compare pre and post-modification
  to decide whether the changes are working as desired or not.
- Not good for testing major changes because we can assume that's from something higher than normal engagement or
  emotional responses causing different behavior.
- Methodology
    - Divide data into A (`Control group`) and B (`Test or variant group`) by random sampling.
    - A remains unchanged but implement change in B.
    - Compare the response from A and B to decide which is better
    - Set null hypothesis as no difference between A and B, and alternative hypothesis making changes in B gives us the
      better result.
    - Calculate number for A and B, and run statistical significance test
        - `Type I error`, rejecting null hypothesis when it is true, meaning accept B when B is not better than A.
        - `Type II error`, accept null hypothesis when it is wrong, meaning reject B when B is actually better than A.
        - `Two-sample T-test`, statistical significance to test whether the average difference between the two groups.
        - Set significance level `alpha` like 0.05.
