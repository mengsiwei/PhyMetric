import React, { useState, useMemo } from "react";
import { MAJORS, AP_CREDITS, DUAL_ENROLLMENT_MAP, MATH_PLACEMENT, resolveAPCredits, resolveMathPlacement } from "../data/majors";
import { getCourse } from "../data/catalog";

const STEPS = ["Major", "Test Scores", "AP Credits", "Transfer Credits", "Review"];

function StepIndicator({ current, total, labels }) {
  return (
    <div className="wizard-steps">
      {labels.map((label, i) => (
        <div key={i} className={`wizard-step ${i === current ? "active" : ""} ${i < current ? "done" : ""}`}>
          <div className="wizard-step-dot">
            {i < current ? "\u2713" : i + 1}
          </div>
          <div className="wizard-step-label">{label}</div>
        </div>
      ))}
    </div>
  );
}

export default function OnboardingWizard({ onComplete }) {
  const [step, setStep] = useState(0);
  const [name, setName] = useState("");
  const [selectedMajor, setSelectedMajor] = useState("cse");

  // Test scores
  const [mathTest, setMathTest] = useState(""); // "ACT Math" or "SAT Math"
  const [mathScore, setMathScore] = useState("");

  // AP exams: array of { examName, score }
  const [apExams, setApExams] = useState([]);
  const [apAdding, setApAdding] = useState({ examName: "", score: "" });

  // Transfer/dual enrollment credits: array of course equivalentIds
  const [transferCredits, setTransferCredits] = useState([]);

  // Computed AP credit grants
  const apGranted = useMemo(() => resolveAPCredits(apExams), [apExams]);
  const mathPlacement = useMemo(
    () => (mathTest && mathScore ? resolveMathPlacement(mathTest, parseInt(mathScore)) : null),
    [mathTest, mathScore]
  );

  // All pre-college credits combined
  const allPreCredits = useMemo(() => {
    const set = new Set([...apGranted, ...transferCredits]);
    return set;
  }, [apGranted, transferCredits]);

  const handleAddAP = () => {
    if (apAdding.examName && apAdding.score) {
      setApExams((prev) => [...prev, { examName: apAdding.examName, score: parseInt(apAdding.score) }]);
      setApAdding({ examName: "", score: "" });
    }
  };

  const handleRemoveAP = (idx) => {
    setApExams((prev) => prev.filter((_, i) => i !== idx));
  };

  const toggleTransfer = (courseId) => {
    setTransferCredits((prev) =>
      prev.includes(courseId) ? prev.filter((id) => id !== courseId) : [...prev, courseId]
    );
  };

  const handleFinish = () => {
    onComplete({
      name: name.trim() || "Student",
      majorId: selectedMajor,
      mathPlacement,
      preCollegeCredits: [...allPreCredits],
      apExams,
      transferCredits,
    });
  };

  const canNext =
    step === 0 ? name.trim() && selectedMajor :
    step === 1 ? true :
    step === 2 ? true :
    step === 3 ? true :
    true;

  return (
    <div className="login-page">
      <div className="wizard-card fade-in">
        <div className="header-logo" style={{ margin: "0 auto 16px", width: 50, height: 50, fontSize: 20 }}>
          CP
        </div>
        <h1 style={{ marginBottom: 4 }}>
          UNR <span>Curriculum Planner</span>
        </h1>
        <p style={{ color: "var(--text-secondary)", fontSize: 13, marginBottom: 20 }}>
          Let's set up your degree plan
        </p>

        <StepIndicator current={step} total={STEPS.length} labels={STEPS} />

        <div className="wizard-body">
          {/* ── Step 0: Name & Major ──────────────────── */}
          {step === 0 && (
            <div className="wizard-section fade-in">
              <label className="wizard-label">Your Name</label>
              <input
                className="login-input"
                placeholder="Enter your name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                autoFocus
              />

              <label className="wizard-label" style={{ marginTop: 16 }}>Select Your Major</label>
              <div className="major-grid">
                {MAJORS.map((m) => (
                  <div
                    key={m.id}
                    className={`major-card ${selectedMajor === m.id ? "selected" : ""}`}
                    onClick={() => setSelectedMajor(m.id)}
                  >
                    <div className="major-card-name">{m.name}</div>
                    <div className="major-card-dept">{m.department}</div>
                    <div className="major-card-desc">{m.description}</div>
                    {m.id !== "cse" && (
                      <div className="major-card-badge">Coming Soon</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── Step 1: Math Test Scores ─────────────── */}
          {step === 1 && (
            <div className="wizard-section fade-in">
              <h3 className="wizard-section-title">Math Placement Test Scores</h3>
              <p className="wizard-hint">
                Your ACT/SAT math score determines which math course you start with.
                This is optional — skip if you don't have scores.
              </p>

              <div className="wizard-row">
                <div className="wizard-field">
                  <label className="wizard-label">Test Type</label>
                  <select
                    className="login-input"
                    value={mathTest}
                    onChange={(e) => setMathTest(e.target.value)}
                  >
                    <option value="">-- Select --</option>
                    <option value="ACT Math">ACT Math</option>
                    <option value="SAT Math">SAT Math</option>
                  </select>
                </div>
                <div className="wizard-field">
                  <label className="wizard-label">Score</label>
                  <input
                    className="login-input"
                    type="number"
                    placeholder={mathTest === "SAT Math" ? "e.g. 640" : "e.g. 28"}
                    value={mathScore}
                    onChange={(e) => setMathScore(e.target.value)}
                  />
                </div>
              </div>

              {mathPlacement && (
                <div className="wizard-result">
                  <strong>{mathPlacement.label}</strong>
                  <span> — You can start with {getCourse(mathPlacement.placesInto)?.code || mathPlacement.placesInto}</span>
                </div>
              )}
              {mathTest && mathScore && !mathPlacement && (
                <div className="wizard-result" style={{ borderColor: "var(--accent-yellow)" }}>
                  Score below placement threshold. You may need a remedial math course first.
                </div>
              )}
            </div>
          )}

          {/* ── Step 2: AP Credits ───────────────────── */}
          {step === 2 && (
            <div className="wizard-section fade-in">
              <h3 className="wizard-section-title">AP Exam Credits</h3>
              <p className="wizard-hint">
                Add any AP exams you've taken. We'll determine which UNR courses you can skip.
              </p>

              <div className="wizard-row" style={{ alignItems: "flex-end" }}>
                <div className="wizard-field" style={{ flex: 2 }}>
                  <label className="wizard-label">AP Exam</label>
                  <select
                    className="login-input"
                    value={apAdding.examName}
                    onChange={(e) => setApAdding((p) => ({ ...p, examName: e.target.value }))}
                  >
                    <option value="">-- Select Exam --</option>
                    {AP_CREDITS.map((ap) => (
                      <option key={ap.examName} value={ap.examName}>
                        {ap.examName}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="wizard-field" style={{ flex: 1 }}>
                  <label className="wizard-label">Score (1-5)</label>
                  <input
                    className="login-input"
                    type="number"
                    min="1"
                    max="5"
                    placeholder="e.g. 4"
                    value={apAdding.score}
                    onChange={(e) => setApAdding((p) => ({ ...p, score: e.target.value }))}
                  />
                </div>
                <button
                  className="btn btn-primary"
                  onClick={handleAddAP}
                  disabled={!apAdding.examName || !apAdding.score}
                  style={{ marginBottom: 0, height: 42 }}
                >
                  Add
                </button>
              </div>

              {apExams.length > 0 && (
                <div className="wizard-items">
                  {apExams.map((ap, i) => (
                    <div key={i} className="wizard-item">
                      <div>
                        <strong>{ap.examName}</strong> — Score: {ap.score}
                      </div>
                      <button className="course-action-btn remove" onClick={() => handleRemoveAP(i)}>
                        &#x2715;
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {apGranted.size > 0 && (
                <div className="wizard-result">
                  <strong>Courses you can skip:</strong>
                  <div className="wizard-grants">
                    {[...apGranted].map((id) => {
                      const c = getCourse(id);
                      return c ? (
                        <span key={id} className="wizard-grant-badge">{c.code} — {c.name}</span>
                      ) : null;
                    })}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── Step 3: Transfer / Dual Enrollment ──── */}
          {step === 3 && (
            <div className="wizard-section fade-in">
              <h3 className="wizard-section-title">Transfer & Dual Enrollment Credits</h3>
              <p className="wizard-hint">
                Select any courses you've already completed through dual enrollment, community college, or transfer.
              </p>

              <div className="transfer-grid">
                {DUAL_ENROLLMENT_MAP.map((item) => {
                  const isSelected = transferCredits.includes(item.equivalentId);
                  const isAPGranted = apGranted.has(item.equivalentId);
                  const course = getCourse(item.equivalentId);
                  return (
                    <div
                      key={item.equivalentId}
                      className={`transfer-item ${isSelected ? "selected" : ""} ${isAPGranted ? "ap-granted" : ""}`}
                      onClick={() => !isAPGranted && toggleTransfer(item.equivalentId)}
                    >
                      <div className="transfer-check">
                        {isSelected || isAPGranted ? "\u2713" : ""}
                      </div>
                      <div className="transfer-info">
                        <div className="transfer-name">{item.transferName}</div>
                        <div className="transfer-equiv">
                          = {course?.code || item.equivalentId}
                          {isAPGranted && <span className="transfer-ap-tag">via AP</span>}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* ── Step 4: Review ───────────────────────── */}
          {step === 4 && (
            <div className="wizard-section fade-in">
              <h3 className="wizard-section-title">Review Your Profile</h3>

              <div className="review-grid">
                <div className="review-card">
                  <div className="review-label">Name</div>
                  <div className="review-value">{name || "Student"}</div>
                </div>
                <div className="review-card">
                  <div className="review-label">Major</div>
                  <div className="review-value">{MAJORS.find((m) => m.id === selectedMajor)?.name}</div>
                </div>
                {mathPlacement && (
                  <div className="review-card">
                    <div className="review-label">Math Placement</div>
                    <div className="review-value">{mathPlacement.label}</div>
                  </div>
                )}
                <div className="review-card">
                  <div className="review-label">Pre-College Credits</div>
                  <div className="review-value">
                    {allPreCredits.size === 0
                      ? "None"
                      : `${allPreCredits.size} course${allPreCredits.size > 1 ? "s" : ""} skipped`}
                  </div>
                </div>
              </div>

              {allPreCredits.size > 0 && (
                <div style={{ marginTop: 16 }}>
                  <div className="wizard-label">Courses you'll skip:</div>
                  <div className="wizard-grants" style={{ marginTop: 8 }}>
                    {[...allPreCredits].map((id) => {
                      const c = getCourse(id);
                      return c ? (
                        <span key={id} className="wizard-grant-badge">{c.code}</span>
                      ) : null;
                    })}
                  </div>
                </div>
              )}

              <p className="wizard-hint" style={{ marginTop: 16 }}>
                You can always modify your plan after setup. Don't worry if you missed something!
              </p>
            </div>
          )}
        </div>

        {/* ── Navigation ─────────────────────────────── */}
        <div className="wizard-nav">
          {step > 0 && (
            <button className="btn btn-secondary" onClick={() => setStep((s) => s - 1)}>
              Back
            </button>
          )}
          <div style={{ flex: 1 }} />
          {step < STEPS.length - 1 ? (
            <button className="btn btn-primary" onClick={() => setStep((s) => s + 1)} disabled={!canNext}>
              Next
            </button>
          ) : (
            <button
              className="login-btn"
              style={{ width: "auto", padding: "10px 32px", marginTop: 0 }}
              onClick={handleFinish}
            >
              Start Planning
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
