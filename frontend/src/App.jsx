import { useState } from 'react'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'

export default function App() {
  const [repoUrl, setRepoUrl] = useState('')
  const [issueNumber, setIssueNumber] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleAnalyze(e) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await fetch(`${BACKEND_URL}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ repo_url: repoUrl.trim(), issue_number: Number(issueNumber) }),
      })

      if (!res.ok) {
        let errorMsg = `Server error ${res.status}`
        try {
          const errData = await res.json()
          if (errData.detail) {
            errorMsg = typeof errData.detail === 'string' ? errData.detail : JSON.stringify(errData.detail)
          }
        } catch {
          const text = await res.text()
          if (text) errorMsg = text
        }
        throw new Error(errorMsg)
      }

      const data = await res.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: 760, margin: '40px auto', fontFamily: 'system-ui, -apple-system, sans-serif', padding: '0 16px', color: '#161616' }}>
      <h1 style={{ marginBottom: 4, fontSize: 28, fontWeight: 700 }}>IssueLens</h1>
      <p style={{ color: '#555', marginTop: 0, marginBottom: 28 }}>
        Deterministic GitHub Issue Code Localization
      </p>

      <form onSubmit={handleAnalyze}>
        <div style={{ marginBottom: 16 }}>
          <label style={{ display: 'block', marginBottom: 6, fontWeight: 600, fontSize: 14 }}>
            Repository URL
          </label>
          <input
            id="repo-url-input"
            type="text"
            value={repoUrl}
            onChange={(e) => setRepoUrl(e.target.value)}
            placeholder="https://github.com/psf/requests"
            required
            style={{ width: '100%', padding: '10px 12px', fontSize: 14, boxSizing: 'border-box', border: '1px solid #ccc', borderRadius: 4 }}
          />
        </div>

        <div style={{ marginBottom: 18 }}>
          <label style={{ display: 'block', marginBottom: 6, fontWeight: 600, fontSize: 14 }}>
            Issue Number
          </label>
          <input
            id="issue-number-input"
            type="number"
            value={issueNumber}
            onChange={(e) => setIssueNumber(e.target.value)}
            placeholder="e.g. 100"
            required
            min={1}
            style={{ width: '100%', padding: '10px 12px', fontSize: 14, boxSizing: 'border-box', border: '1px solid #ccc', borderRadius: 4 }}
          />
        </div>

        <button
          id="analyze-submit-button"
          type="submit"
          disabled={loading}
          style={{
            padding: '10px 24px',
            fontSize: 14,
            fontWeight: 600,
            background: loading ? '#8d8d8d' : '#0f62fe',
            color: '#fff',
            border: 'none',
            borderRadius: 4,
            cursor: loading ? 'not-allowed' : 'pointer',
          }}
        >
          {loading ? 'Analyzing Issue…' : 'Analyze Issue'}
        </button>
      </form>

      {loading && (
        <div style={{ marginTop: 24, padding: '14px 18px', background: '#edf5ff', border: '1px solid #c9deff', borderRadius: 4, color: '#0043ce' }}>
          <strong>Running localization pipeline:</strong> Fetching issue and repository AST, scoring candidates, tracing execution paths, and identifying relevant tests…
        </div>
      )}

      {error && (
        <div style={{ marginTop: 24, padding: '14px 18px', background: '#fff1f1', border: '1px solid #ffd7d9', borderRadius: 4, color: '#da1e28' }}>
          <strong>Analysis Error:</strong> {error}
        </div>
      )}

      {result && (
        <div style={{ marginTop: 28, display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Header Info */}
          <div style={{ padding: '16px 20px', background: '#f4f4f4', border: '1px solid #e0e0e0', borderRadius: 4 }}>
            <div style={{ fontSize: 13, color: '#6f6f6f', textTransform: 'uppercase', fontWeight: 600, letterSpacing: 0.5 }}>
              Issue #{result.issue_number}
            </div>
            <h2 style={{ margin: '4px 0 8px 0', fontSize: 18, fontWeight: 600 }}>{result.issue_title}</h2>
            <div style={{ fontSize: 13, color: '#525252' }}>
              Analyzed {result.total_files_analyzed} repository files
            </div>
          </div>

          {/* Uncertainty banner if applicable */}
          {result.is_uncertain ? (
            <div style={{ padding: '16px 20px', background: '#fff8e1', border: '1px solid #ffe082', borderRadius: 4, color: '#b78103' }}>
              <h3 style={{ margin: '0 0 6px 0', fontSize: 16 }}>Localization Uncertain</h3>
              <p style={{ margin: 0, fontSize: 14 }}>
                {result.uncertainty_detail || `Unable to name a confident primary location (${result.uncertainty_reason}).`}
              </p>
            </div>
          ) : (
            /* Primary Localization Card */
            <div style={{ padding: '18px 20px', background: '#f4fbf6', border: '1px solid #a7f3d0', borderRadius: 4 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <span style={{ fontSize: 12, fontWeight: 700, color: '#0e6245', textTransform: 'uppercase', letterSpacing: 0.5 }}>
                  Primary Location
                </span>
                <span style={{ fontSize: 13, fontWeight: 600, background: '#d1fae5', color: '#065f46', padding: '2px 8px', borderRadius: 12 }}>
                  Score: {(result.evidence_score * 100).toFixed(0)}% ({result.evidence_confidence || 'moderate'})
                </span>
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ fontSize: 13, color: '#525252' }}>File:</div>
                <code style={{ fontSize: 15, fontWeight: 700, color: '#047857' }}>{result.primary_file}</code>
              </div>

              {result.primary_symbol && (
                <div style={{ marginBottom: 10 }}>
                  <div style={{ fontSize: 13, color: '#525252' }}>Function / Class:</div>
                  <code style={{ fontSize: 14, fontWeight: 600, color: '#1f2937' }}>
                    {result.primary_symbol} {result.primary_symbol_kind ? `(${result.primary_symbol_kind})` : ''}
                  </code>
                </div>
              )}

              {result.line_range && (
                <div style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: 13, color: '#525252' }}>Line Range:</div>
                  <span style={{ fontSize: 14, fontWeight: 600, color: '#374151' }}>
                    Lines {result.line_range.start} – {result.line_range.end}
                  </span>
                </div>
              )}

              <div>
                <div style={{ fontSize: 13, color: '#525252', marginBottom: 4 }}>Explanation:</div>
                <p style={{ margin: 0, fontSize: 14, lineHeight: 1.5, color: '#1f2937' }}>
                  {result.explanation}
                </p>
              </div>

              {result.ai_explanation && (
                <div style={{ marginTop: 14, paddingTop: 12, borderTop: '1px dashed #a7f3d0' }}>
                  <div style={{ fontSize: 12, fontWeight: 700, color: '#065f46', textTransform: 'uppercase', marginBottom: 4 }}>
                    AI Explanation:
                  </div>
                  <p style={{ margin: 0, fontSize: 13.5, lineHeight: 1.5, color: '#27272a', whiteSpace: 'pre-line' }}>
                    {result.ai_explanation}
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Execution Path */}
          {result.execution_path && result.execution_path.length > 0 && (
            <div style={{ padding: '16px 20px', background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 4 }}>
              <h3 style={{ margin: '0 0 10px 0', fontSize: 15, fontWeight: 600, color: '#1e293b' }}>
                Execution Path
              </h3>
              <ol style={{ margin: 0, paddingLeft: 20, fontSize: 13.5, lineHeight: 1.6, color: '#334155' }}>
                {result.execution_path.map((step, idx) => (
                  <li key={idx} style={{ marginBottom: 4 }}>{step}</li>
                ))}
              </ol>
            </div>
          )}

          {/* Relevant Tests */}
          {result.relevant_tests && result.relevant_tests.length > 0 && (
            <div style={{ padding: '16px 20px', background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 4 }}>
              <h3 style={{ margin: '0 0 10px 0', fontSize: 15, fontWeight: 600, color: '#1e293b' }}>
                Relevant Tests
              </h3>
              <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13.5, lineHeight: 1.6, color: '#334155' }}>
                {result.relevant_tests.map((test, idx) => (
                  <li key={idx} style={{ marginBottom: 6 }}>
                    <strong>{test.path}</strong>: {test.reason}
                    {test.test_symbols && test.test_symbols.length > 0 && (
                      <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>
                        Test functions: {test.test_symbols.join(', ')}
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Related Files */}
          {result.related_files && result.related_files.length > 0 && (
            <div style={{ padding: '16px 20px', background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 4 }}>
              <h3 style={{ margin: '0 0 10px 0', fontSize: 15, fontWeight: 600, color: '#1e293b' }}>
                Related Files
              </h3>
              <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13.5, lineHeight: 1.6, color: '#334155' }}>
                {result.related_files.map((rf, idx) => (
                  <li key={idx} style={{ marginBottom: 4 }}>
                    <strong>{rf.path}</strong> {rf.symbol ? `(${rf.symbol})` : ''} — {rf.reason} (score: {rf.score.toFixed(1)})
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
