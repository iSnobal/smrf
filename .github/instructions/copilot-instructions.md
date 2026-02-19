When reviewing code, focus on:

## Security Critical Issues
- Check for hardcoded secrets, API keys, or credentials

## Performance Red Flags
- Spot inefficient loops and algorithmic issues
- Suggest converting code to faster or compiled alternatives for large datasets
- Check for memory leaks and resource cleanup
- Review caching opportunities for expensive operations

## Code Quality Essentials
- Use clear, descriptive naming conventions
- Ensure basic error handling
- Do not suggest additional or excessive commenting of code
- Review tests if they could potentially be unstable or flaky

## Review Style
- Be short and concise
- Explain the "why" behind recommendations
- Ask clarifying questions when code intent is unclear
- Do not repeat comments that were previously resolved on new pushes

## Review Summary
- Do not repeat any information that was already in the PR description
- Focus on logic over technical changes
