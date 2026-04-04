import React, { useState, useEffect, useCallback } from "react";
import OnboardingWizard from "./components/OnboardingWizard";
import Sidebar from "./components/Sidebar";
import SemesterTimeline from "./components/SemesterTimeline";
import SkillTree from "./components/SkillTree";
import ProgressDashboard from "./components/ProgressDashboard";
import CourseDetailModal from "./components/CourseDetailModal";
import CoursePickerModal from "./components/CoursePickerModal";
import IssuesPanel from "./components/IssuesPanel";
import Milestones from "./components/Milestones";
import AIChatBox from "./components/AIChatBox";
import { defaultPlan, semesterLabel, getCourse, UNIVERSITY } from "./data/catalog";
import { validatePlan, computeProgress, cascadeRemove, autoSchedule } from "./utils/planEngine";
import { MAJORS } from "./data/majors";

const STORAGE_KEY = "unr_curriculum_planner";

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch {}
  return null;
}

function saveState(state) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

/**
 * Build an initial plan that removes pre-college credits from the default plan
 * and re-schedules remaining courses to fill gaps.
 */
function buildInitialPlan(preCollegeCredits) {
  if (!preCollegeCredits || preCollegeCredits.length === 0) return defaultPlan;

  const skipSet = new Set(preCollegeCredits);

  // Remove skipped courses from the default plan
  let plan = defaultPlan.map((sem) => sem.filter((id) => !skipSet.has(id)));

  // Compact: move courses forward to fill empty slots (keep max 5 per semester)
  const flat = [];
  plan.forEach((sem, semIdx) => {
    sem.forEach((id) => flat.push({ id, originalSem: semIdx }));
  });

  const result = Array.from({ length: 8 }, () => []);
  for (const { id, originalSem } of flat) {
    // Try to place in the same or earlier semester
    let placed = false;
    for (let s = 0; s <= originalSem; s++) {
      if (result[s].length < 5) {
        result[s].push(id);
        placed = true;
        break;
      }
    }
    if (!placed) {
      // Fall back to original semester or later
      for (let s = originalSem; s < 8; s++) {
        if (result[s].length < 5) {
          result[s].push(id);
          placed = true;
          break;
        }
      }
      if (!placed) result[7].push(id);
    }
  }

  return result;
}

export default function App() {
  const [user, setUser] = useState(null);
  const [plan, setPlan] = useState(defaultPlan);
  const [completedSemesters, setCompletedSemesters] = useState(0);
  const [preCollegeCredits, setPreCollegeCredits] = useState([]);
  const [view, setView] = useState("timeline");
  const [selectedCourse, setSelectedCourse] = useState(null);
  const [pickerSemester, setPickerSemester] = useState(null);

  // Load saved state
  useEffect(() => {
    const saved = loadState();
    if (saved) {
      if (saved.user) setUser(saved.user);
      if (saved.plan) setPlan(saved.plan);
      if (saved.completedSemesters != null) setCompletedSemesters(saved.completedSemesters);
      if (saved.preCollegeCredits) setPreCollegeCredits(saved.preCollegeCredits);
    }
  }, []);

  // Save on changes
  useEffect(() => {
    if (user) {
      saveState({ user, plan, completedSemesters, preCollegeCredits });
    }
  }, [user, plan, completedSemesters, preCollegeCredits]);

  const issues = validatePlan(plan);
  const progress = computeProgress(plan, completedSemesters);

  /** Called when onboarding wizard completes */
  const handleOnboardingComplete = useCallback((profile) => {
    const userObj = {
      name: profile.name,
      majorId: profile.majorId,
      majorName: MAJORS.find((m) => m.id === profile.majorId)?.name || profile.majorId,
      mathPlacement: profile.mathPlacement,
    };
    setUser(userObj);
    setPreCollegeCredits(profile.preCollegeCredits);

    // Build a plan that accounts for pre-college credits
    const initialPlan = buildInitialPlan(profile.preCollegeCredits);
    setPlan(initialPlan);

    // If they have pre-college credits, mark them as "pre-completed"
    // We won't mark any semester as completed since they haven't started yet,
    // but the credits are simply removed from the plan.
  }, []);

  const handleLogout = useCallback(() => {
    localStorage.removeItem(STORAGE_KEY);
    setUser(null);
    setPlan(defaultPlan);
    setCompletedSemesters(0);
    setPreCollegeCredits([]);
  }, []);

  const handleRemoveCourse = useCallback((semIdx, courseId) => {
    setPlan((prev) => cascadeRemove(courseId, prev));
  }, []);

  const handleAddCourse = useCallback((semIdx, courseId) => {
    setPlan((prev) => {
      const newPlan = prev.map((s) => [...s]);
      if (!newPlan[semIdx].includes(courseId)) {
        newPlan[semIdx].push(courseId);
      }
      return newPlan;
    });
    setPickerSemester(null);
  }, []);

  const handleSubstitute = useCallback((oldId, newId, semIdx) => {
    setPlan((prev) => {
      const newPlan = prev.map((s) => [...s]);
      const idx = newPlan[semIdx].indexOf(oldId);
      if (idx !== -1) {
        newPlan[semIdx][idx] = newId;
      }
      return newPlan;
    });
    setSelectedCourse(null);
  }, []);

  const handleSwapCourse = useCallback((oldId, newId, semIdx) => {
    setPlan((prev) => {
      const newPlan = prev.map((s) => [...s]);
      let targetSem = semIdx;
      for (let i = 0; i < newPlan.length; i++) {
        if (newPlan[i].includes(oldId)) {
          targetSem = i;
          break;
        }
      }
      const idx = newPlan[targetSem].indexOf(oldId);
      if (idx !== -1) {
        newPlan[targetSem][idx] = newId;
      } else {
        if (!newPlan[targetSem].includes(newId)) {
          newPlan[targetSem].push(newId);
        }
      }
      return newPlan;
    });
  }, []);

  const handleToggleSemesterComplete = useCallback((semIdx) => {
    setCompletedSemesters((prev) => (semIdx < prev ? semIdx : semIdx + 1));
  }, []);

  const handleAutoSchedule = useCallback(() => {
    const completed = [...preCollegeCredits];
    for (let i = 0; i < completedSemesters; i++) {
      plan[i].forEach((id) => completed.push(id));
    }
    const newPlan = autoSchedule(completed, 8);
    const final = plan.map((s) => [...s]);
    for (let i = completedSemesters; i < 8; i++) {
      final[i] = newPlan[i] || [];
    }
    setPlan(final);
  }, [completedSemesters, plan, preCollegeCredits]);

  const handleReset = useCallback(() => {
    setPlan(buildInitialPlan(preCollegeCredits));
    setCompletedSemesters(0);
  }, [preCollegeCredits]);

  const handleAddSemester = useCallback(() => {
    setPlan((prev) => [...prev, []]);
  }, []);

  // ── Show onboarding wizard if no user ──
  if (!user) {
    return <OnboardingWizard onComplete={handleOnboardingComplete} />;
  }

  // Flatten issues for display
  const allIssues = [];
  issues.forEach((semIssues, semIdx) => {
    Object.entries(semIssues).forEach(([courseId, msgs]) => {
      msgs.forEach((msg) => {
        const course = getCourse(courseId);
        allIssues.push({
          semester: semesterLabel(semIdx),
          course: course?.code || courseId,
          message: msg,
        });
      });
    });
  });

  return (
    <div className="app-container">
      <header className="header">
        <div className="header-left">
          <div className="header-logo">CP</div>
          <h1>
            UNR <span>Curriculum Planner</span>
          </h1>
        </div>
        <div className="header-right">
          <div className="header-user">
            <div className="user-avatar">{user.name.charAt(0).toUpperCase()}</div>
            <div>
              <div>{user.name}</div>
              <div style={{ fontSize: 10, color: "var(--text-muted)" }}>{user.majorName}</div>
            </div>
          </div>
          <button className="btn btn-secondary" onClick={handleLogout}>
            Logout
          </button>
        </div>
      </header>

      <div className="main-content">
        <Sidebar
          view={view}
          setView={setView}
          progress={progress}
          completedSemesters={completedSemesters}
          totalSemesters={plan.length}
          preCollegeCredits={preCollegeCredits}
        />

        <div className="content-area">
          <h2 className="page-title">
            {view === "timeline" && "Semester Timeline"}
            {view === "skilltree" && "Prerequisite Skill Tree"}
            {view === "progress" && "Degree Progress"}
          </h2>
          <p className="page-subtitle">{user.majorName} — {UNIVERSITY}</p>

          {/* Pre-college credits banner */}
          {preCollegeCredits.length > 0 && view === "timeline" && (
            <div className="precollege-banner">
              <strong>Pre-college credits applied:</strong>{" "}
              {preCollegeCredits.map((id) => getCourse(id)?.code || id).join(", ")}
              <span className="precollege-count">({preCollegeCredits.length} courses skipped)</span>
            </div>
          )}

          {view === "timeline" && (
            <>
              <div className="btn-group">
                <button className="btn btn-primary" onClick={handleAutoSchedule}>
                  Auto-Schedule
                </button>
                <button className="btn btn-secondary" onClick={handleReset}>
                  Reset Plan
                </button>
                <button className="btn btn-secondary" onClick={handleAddSemester}>
                  + Add Semester
                </button>
              </div>

              {allIssues.length > 0 && <IssuesPanel issues={allIssues} />}

              <Milestones progress={progress} completedSemesters={completedSemesters} />

              <SemesterTimeline
                plan={plan}
                issues={issues}
                completedSemesters={completedSemesters}
                onCourseClick={(courseId, semIdx) => setSelectedCourse({ courseId, semIdx })}
                onRemoveCourse={handleRemoveCourse}
                onAddCourse={(semIdx) => setPickerSemester(semIdx)}
                onToggleComplete={handleToggleSemesterComplete}
              />
            </>
          )}

          {view === "skilltree" && (
            <SkillTree
              plan={plan}
              completedSemesters={completedSemesters}
              preCollegeCredits={preCollegeCredits}
              onCourseClick={(courseId) => setSelectedCourse({ courseId, semIdx: 0 })}
            />
          )}

          {view === "progress" && (
            <ProgressDashboard
              progress={progress}
              plan={plan}
              completedSemesters={completedSemesters}
              preCollegeCredits={preCollegeCredits}
            />
          )}
        </div>
      </div>

      {selectedCourse && (
        <CourseDetailModal
          courseId={selectedCourse.courseId}
          semesterIndex={selectedCourse.semIdx}
          plan={plan}
          onClose={() => setSelectedCourse(null)}
          onSubstitute={handleSubstitute}
        />
      )}

      {pickerSemester !== null && (
        <CoursePickerModal
          semesterIndex={pickerSemester}
          plan={plan}
          onAdd={handleAddCourse}
          onClose={() => setPickerSemester(null)}
        />
      )}

      <AIChatBox
        plan={plan}
        completedSemesters={completedSemesters}
        preCollegeCredits={preCollegeCredits}
        onAddCourse={handleAddCourse}
        onSwapCourse={handleSwapCourse}
      />
    </div>
  );
}
