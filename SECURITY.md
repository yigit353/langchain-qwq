# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| < 0.3   | :x:                |

## Reporting a Vulnerability

We take the security of langchain-qwq seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Reporting Process

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to:
- **yigit353@gmail.com**
- **tiebingice123@outlook.com**

Please include the following information in your report:

1. **Type of issue** (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
2. **Full paths of source file(s)** related to the manifestation of the issue
3. **The location of the affected source code** (tag/branch/commit or direct URL)
4. **Any special configuration** required to reproduce the issue
5. **Step-by-step instructions** to reproduce the issue
6. **Proof-of-concept or exploit code** (if possible)
7. **Impact of the issue**, including how an attacker might exploit it

This information will help us triage your report more quickly.

### Response Timeline

- **Initial Response:** Within 48 hours
- **Status Update:** Within 5 business days
- **Resolution Timeline:** Varies based on severity and complexity

### What to Expect

After submitting a report, you can expect:

1. **Acknowledgment:** We will acknowledge receipt of your vulnerability report within 48 hours
2. **Communication:** We will keep you informed about our progress toward a fix
3. **Verification:** We may ask for additional information or guidance
4. **Credit:** We will credit you in the security advisory (unless you prefer to remain anonymous)
5. **Disclosure:** We will coordinate disclosure with you

### Security Update Process

When we receive a security bug report, we will:

1. Confirm the problem and determine the affected versions
2. Audit code to find any similar problems
3. Prepare fixes for all supported versions
4. Release new security patch versions as soon as possible

### Security Best Practices for Users

#### API Key Security

1. **Never commit API keys** to version control
2. **Use environment variables** for API key storage:
   ```bash
   export DASHSCOPE_API_KEY="your-api-key-here"
   ```
3. **Rotate keys regularly** as part of your security hygiene
4. **Use separate keys** for development, testing, and production
5. **Implement key rotation** policies in production environments

#### Secure Configuration

1. **Set appropriate timeouts:**
   ```python
   from langchain_qwq import ChatQwQ

   model = ChatQwQ(
       model="qwq-plus",
       timeout=30,  # Set reasonable timeout
       max_retries=2  # Limit retry attempts
   )
   ```

2. **Validate and sanitize inputs:**
   ```python
   # Validate message length before sending
   MAX_MESSAGE_LENGTH = 100000
   if len(message) > MAX_MESSAGE_LENGTH:
       raise ValueError("Message too long")
   ```

3. **Handle exceptions properly:**
   ```python
   import logging

   try:
       response = model.invoke(message)
   except Exception as e:
       # Log error without exposing sensitive data
       logging.error("API call failed: %s", type(e).__name__)
       # Don't log the full exception which might contain API keys
   ```

#### Dependency Security

1. **Keep dependencies updated:**
   ```bash
   pip install --upgrade langchain-qwq
   ```

2. **Use dependency scanning:**
   ```bash
   pip install pip-audit
   pip-audit
   ```

3. **Pin your dependencies** in production:
   ```bash
   pip freeze > requirements.txt
   ```

#### Network Security

1. **Use HTTPS only** (library enforces this by default)
2. **Validate SSL certificates** (enabled by default)
3. **Use firewall rules** to restrict outbound connections if needed
4. **Monitor API usage** for unusual patterns

#### Production Deployment

1. **Use secrets management** systems (AWS Secrets Manager, Azure Key Vault, HashiCorp Vault)
2. **Implement rate limiting** at application level
3. **Enable comprehensive logging** (without logging sensitive data)
4. **Monitor for security events** and anomalies
5. **Regular security audits** of your implementation

### Known Security Considerations

#### 1. API Key Handling

The library uses Pydantic's `SecretStr` to protect API keys in memory, but:
- Keys are still accessible in the process memory
- Keys appear in environment variables
- Ensure your deployment environment is secure

#### 2. Input Validation

While the library validates data using Pydantic models:
- Always validate user inputs at the application level
- Implement rate limiting for user-facing applications
- Set reasonable message size limits

#### 3. Error Messages

Error messages may contain implementation details:
- Implement error sanitization in production
- Don't expose raw error messages to end users
- Log detailed errors securely for debugging

#### 4. Concurrent Usage

When using the library in multi-threaded applications:
- Each thread should have its own model instance
- Be aware of global state modifications (see audit report)

### Security Features

✅ **Implemented:**
- Environment-based API key loading
- SecretStr for in-memory key protection
- HTTPS-only API communication
- Type-safe implementations with mypy
- No use of dangerous functions (eval, exec, etc.)

⚠️ **User Responsibility:**
- Rate limiting
- Input validation
- Error message sanitization
- Timeout configuration
- Monitoring and logging

### Vulnerability Disclosure Policy

We follow coordinated vulnerability disclosure:

1. Security researchers privately report vulnerabilities
2. We work with reporters to understand and validate issues
3. We develop and test fixes
4. We release security updates
5. We publicly disclose the vulnerability after fixes are available

### Security Hardening Checklist

Before deploying to production:

- [ ] API keys stored in secure secrets management system
- [ ] Environment variables are not logged or exposed
- [ ] Input validation implemented at application level
- [ ] Rate limiting configured appropriately
- [ ] Timeouts set to reasonable values
- [ ] Error handling sanitizes sensitive information
- [ ] Dependencies are up to date and scanned for vulnerabilities
- [ ] Monitoring and alerting configured
- [ ] Security logging enabled
- [ ] Access controls implemented
- [ ] Regular security updates scheduled

### Additional Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)
- [LangChain Security Documentation](https://python.langchain.com/docs/security)

### Contact

For security-related questions or concerns:
- Email: yigit353@gmail.com, tiebingice123@outlook.com
- GitHub Issues: For non-security bugs and feature requests only

### Attribution

We appreciate the security research community and will acknowledge researchers who report valid security issues (with their permission).

---

**Last Updated:** 2025-11-16
