import React, { useState, useEffect, useRef } from 'react';
import confetti from 'canvas-confetti';
import { 
  CheckCircle2, 
  Camera, 
  RefreshCw, 
  ArrowRight, 
  ArrowLeft,
  Timer, 
  AlertTriangle,
  UserPlus,
  X,
  ShieldCheck,
  Lock,
  Cpu,
  Eye,
  Check,
  Info,
  Scan,
  Hand,
  Sparkles,
  Trash2
} from 'lucide-react';

type AppState = 'idle' | 'scan' | 'enroll';
type CameraState = 'connecting' | 'live' | 'error';

interface ScanResult {
  accepted: boolean;
  username: string | null;
  score: number;
  threshold: number;
  time_ms: number;
  clahe_base64?: string;
}

interface ReportData {
  users?: Array<{ id: number; username: string; sample_count: number; created_at: string }>;
  total_templates?: number;
}

// ── Soft, Simple, Minimal Palm Guide Overlay (Camera Viewports) ──
// Anatomical smooth palm outline: accommodates BOTH left and right hands cleanly.
// NO text over video feed so user clearly sees their actual hand position!
function SoftPalmGuide() {
  return (
    <div className="absolute inset-0 pointer-events-none flex items-center justify-center p-3 z-10 select-none">
      <div className="relative w-52 h-64 sm:w-60 sm:h-72 flex items-center justify-center">
        
        {/* Soft, minimal camera viewfinder corner brackets */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-0 left-0 w-6 h-6 border-t-[2.5px] border-l-[2.5px] border-white/45 rounded-tl-xl" />
          <div className="absolute top-0 right-0 w-6 h-6 border-t-[2.5px] border-r-[2.5px] border-white/45 rounded-tr-xl" />
          <div className="absolute bottom-0 left-0 w-6 h-6 border-b-[2.5px] border-l-[2.5px] border-white/45 rounded-bl-xl" />
          <div className="absolute bottom-0 right-0 w-6 h-6 border-b-[2.5px] border-r-[2.5px] border-white/45 rounded-br-xl" />
        </div>

        {/* Anatomically natural, soft minimal palm guide (works for both hands) */}
        <svg 
          viewBox="0 0 160 200" 
          className="w-36 h-44 sm:w-44 sm:h-54 drop-shadow-[0_2px_4px_rgba(0,0,0,0.85)]"
          fill="none"
        >
          <path 
            d="M 65 185 C 55 170, 48 150, 45 135 C 40 125, 25 115, 24 102 C 23 93, 33 88, 40 96 C 46 103, 50 110, 54 114 C 53 95, 53 65, 55 46 C 56 36, 68 36, 69 46 C 70 60, 71 72, 71 78 C 72 60, 74 38, 77 26 C 79 16, 91 16, 93 26 C 95 40, 96 62, 96 76 C 97 62, 100 45, 103 36 C 105 27, 117 28, 118 38 C 119 50, 119 68, 119 82 C 121 72, 125 58, 128 52 C 130 44, 140 47, 140 56 C 140 68, 137 92, 136 108 C 134 130, 125 155, 115 170 C 108 180, 102 185, 95 185" 
            stroke="rgba(255,255,255,0.42)" 
            strokeWidth="1.8" 
            strokeDasharray="4 4" 
            strokeLinecap="round" 
            strokeLinejoin="round" 
            fill="rgba(255,255,255,0.02)" 
          />
          {/* Subtle center target reticle where the vascular sensor reads */}
          <circle cx="88" cy="120" r="15" stroke="rgba(255,222,89,0.5)" strokeWidth="1.5" strokeDasharray="3 3" />
          <circle cx="88" cy="120" r="2.5" fill="rgba(255,222,89,0.7)" />
        </svg>

      </div>
    </div>
  );
}

// ── Camera Viewport (Crystal Clear User View - ZERO text/badges over video) ──
function CameraViewport({
  cameraState,
  cameraErrorDetail,
  onRetry,
  children,
  className = "w-full h-full min-h-[300px]",
}: {
  cameraState: CameraState;
  cameraErrorDetail?: string;
  onRetry?: () => void;
  children?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={`relative rounded-2xl border-[3.5px] border-black shadow-[4px_4px_0px_#121212] overflow-hidden bg-black flex items-center justify-center ${className}`}>
      {/* STATE 1: CONNECTING */}
      {cameraState === 'connecting' && (
        <div className="flex flex-col items-center justify-center text-center p-5 text-white animate-pulse space-y-2.5">
          <div className="w-12 h-12 rounded-2xl bg-[#1e1e1e] border-[2px] border-[#FFDE59] flex items-center justify-center">
            <RefreshCw className="w-6 h-6 animate-spin text-[#FFDE59]" />
          </div>
          <span className="font-display font-black text-xs uppercase tracking-wider text-[#FFDE59]">
            CONNECTING TO SENSOR...
          </span>
        </div>
      )}

      {/* STATE 2: LIVE FEED - Pure video feed, crystal clear without cropping */}
      {cameraState === 'live' && (
        <img 
          src="/api/video_feed" 
          alt="Live Camera Feed" 
          className="w-full h-full object-contain bg-black"
        />
      )}

      {/* STATE 3: ERROR / OFFLINE */}
      {cameraState === 'error' && (
        <div className="flex flex-col items-center justify-center text-center p-5 bg-[#250404] text-white w-full h-full space-y-2">
          <div className="w-11 h-11 rounded-2xl bg-[#FF4081] border-[2px] border-black flex items-center justify-center shadow-[2px_2px_0px_#121212]">
            <AlertTriangle className="w-6 h-6 text-white" />
          </div>
          <span className="font-display font-black text-xs uppercase tracking-wider text-[#FF4081]">
            CAMERA SENSOR OFFLINE
          </span>
          <p className="text-[10px] font-bold text-[#FFB0B0] max-w-[240px] leading-tight">
            {cameraErrorDetail || "Camera not detected — check camera cable on Raspberry Pi."}
          </p>
          {onRetry && (
            <button
              onClick={onRetry}
              className="mt-1 px-3 py-1.5 bg-[#FFDE59] text-black border-[2px] border-black rounded-xl text-[11px] font-black shadow-[2px_2px_0px_#121212] neo-btn flex items-center gap-1 cursor-pointer"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Retry Connection</span>
            </button>
          )}
        </div>
      )}

      {children}
    </div>
  );
}

// ── 6 Dynamic Sample Instructions for Enrollment ──
const ENROLL_SAMPLE_INSTRUCTIONS = [
  "Sample 1: Keep palm flat and centered over sensor",
  "Sample 2: Tilt palm slightly to the left",
  "Sample 3: Tilt palm slightly to the right",
  "Sample 4: Tilt palm slightly upwards",
  "Sample 5: Tilt palm slightly downwards",
  "Sample 6: Hold palm flat for final calibration",
];

// ── Error Boundary ──
class ErrorBoundary extends React.Component<{ children: React.ReactNode }, { hasError: boolean }> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }
  static getDerivedStateFromError() {
    return { hasError: true };
  }
  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('ErrorBoundary caught:', error, errorInfo);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="flex-1 flex flex-col items-center justify-center p-6 text-center bg-[#250404] text-white">
          <AlertTriangle className="w-12 h-12 text-[#FF4081] mb-3" />
          <h2 className="font-display font-black text-lg">System Notice</h2>
          <p className="text-xs text-[#FFB0B0] mt-2 max-w-[280px]">The kiosk interface encountered an unexpected state. Please tap to restart.</p>
          <button 
            onClick={() => window.location.reload()} 
            className="mt-4 px-4 py-2 bg-[#FFDE59] text-black border-[2px] border-black rounded-xl text-xs font-black shadow-[2px_2px_0px_#121212]"
          >
            Restart Kiosk
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  // 3-Screen Public State Machine (supports ?screen=scan or ?screen=enroll)
  const [appState, setAppState] = useState<AppState>(() => {
    try {
      const param = new URLSearchParams(window.location.search).get('screen');
      if (param === 'scan' || param === 'enroll' || param === 'idle') return param;
    } catch {}
    return 'idle';
  });

  // Hardware Status
  const [cameraState, setCameraState] = useState<CameraState>('connecting');
  const [cameraErrorDetail, setCameraErrorDetail] = useState<string>('');
  const [cameraReady, setCameraReady] = useState(false);
  const [totalUsers, setTotalUsers] = useState(0);
  const [modelLoaded, setModelLoaded] = useState(false);

  // Scanning State
  const [isScanning, setIsScanning] = useState(false);
  const [scanCountdown, setScanCountdown] = useState<number | null>(null);
  const [resultOverlay, setResultOverlay] = useState<ScanResult | null>(null);

  // Enrollment State (3-6 Samples)
  const [enrollUsername, setEnrollUsername] = useState('');
  const [enrollSamples, setEnrollSamples] = useState<Array<{ vr_mean: number; thumb: string }>>([]);
  const [isCapturingSample, setIsCapturingSample] = useState(false);
  const [enrollCountdown, setEnrollCountdown] = useState<number | null>(null);
  const [enrollStatusMsg, setEnrollStatusMsg] = useState('');

  // Hidden Admin / Ops Modal (5-tap trigger on status bead)
  const [adminOpen, setAdminOpen] = useState(false);
  const [adminTapCount, setAdminTapCount] = useState(0);
  const [reportData, setReportData] = useState<ReportData | null>(null);
  const [reportLoading, setReportLoading] = useState(false);

  // Active state ref to cancel async countdowns on navigation
  const appStateRef = useRef<AppState>(appState);
  useEffect(() => {
    appStateRef.current = appState;
  }, [appState]);

  // Toast Notification
  const [toast, setToast] = useState<{ msg: string; type: 'success' | 'warn' | 'error' } | null>(null);

  const showToast = (msg: string, type: 'success' | 'warn' | 'error' = 'success') => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3500);
  };

  // Poll Hardware Status
  const loadStatus = async () => {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        const available = Boolean(data.camera_available);
        setCameraReady(available);
        setTotalUsers(data.enrolled_users_count || 0);
        setModelLoaded(data.model_loaded || false);

        if (available && data.camera_type !== 'None') {
          setCameraState('live');
          setCameraErrorDetail('');
        } else {
          setCameraState('error');
          setCameraErrorDetail('Camera offline — verify cable connection.');
        }
      } else {
        setCameraReady(false);
        setModelLoaded(false);
        setCameraState('error');
        setCameraErrorDetail('Server unavailable.');
      }
    } catch {
      setCameraReady(false);
      setModelLoaded(false);
      setCameraState('error');
      setCameraErrorDetail('Cannot reach backend server.');
    }
  };

  useEffect(() => {
    loadStatus();
    const interval = setInterval(loadStatus, 4000);
    return () => clearInterval(interval);
  }, []);

  // Timers
  const resultAutoReturnTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const idleTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearResultAutoReturn = () => {
    if (resultAutoReturnTimerRef.current) {
      clearTimeout(resultAutoReturnTimerRef.current);
      resultAutoReturnTimerRef.current = null;
    }
  };

  const closeResultOverlay = (target: 'scan' | 'idle' = 'scan') => {
    clearResultAutoReturn();
    setResultOverlay(null);
    if (target === 'idle') {
      setAppState('idle');
    }
  };

  // Ensure timers are cleared whenever appState changes
  useEffect(() => {
    return () => {
      clearResultAutoReturn();
      if (idleTimeoutRef.current) clearTimeout(idleTimeoutRef.current);
    };
  }, [appState]);

  // Cancel enrollment when leaving enroll state with partial samples
  useEffect(() => {
    if (appState !== 'enroll' && enrollSamples.length > 0 && enrollUsername) {
      fetch('/api/enroll/cancel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: enrollUsername }),
      }).catch(() => {});
      setEnrollSamples([]);
      setEnrollUsername('');
      setEnrollStatusMsg('');
    }
  }, [appState]);

  // Idle Timeout: if Scan screen has no interaction for 35s, return to Idle
  useEffect(() => {
    if (appState !== 'scan') return;
    if (isScanning || scanCountdown !== null || resultOverlay !== null) return;

    const startIdleCountdown = () => {
      if (idleTimeoutRef.current) clearTimeout(idleTimeoutRef.current);
      idleTimeoutRef.current = setTimeout(() => {
        if (appStateRef.current === 'scan') {
          setAppState('idle');
        }
      }, 35000);
    };

    startIdleCountdown();
    const resetOnUserActivity = () => startIdleCountdown();

    window.addEventListener('touchstart', resetOnUserActivity, { passive: true });
    window.addEventListener('mousedown', resetOnUserActivity, { passive: true });

    return () => {
      if (idleTimeoutRef.current) clearTimeout(idleTimeoutRef.current);
      window.removeEventListener('touchstart', resetOnUserActivity);
      window.removeEventListener('mousedown', resetOnUserActivity);
    };
  }, [appState, isScanning, scanCountdown, resultOverlay]);

  // Scan Execution with 3-Second Countdown
  const handleScanWithCountdown = async () => {
    if (isScanning || scanCountdown !== null || !cameraReady) return;
    clearResultAutoReturn();
    
    // 3-Second countdown
    for (let i = 3; i > 0; i--) {
      if (appStateRef.current !== 'scan') {
        setScanCountdown(null);
        return;
      }
      setScanCountdown(i);
      await new Promise(r => setTimeout(r, 900));
    }
    if (appStateRef.current !== 'scan') {
      setScanCountdown(null);
      return;
    }
    setScanCountdown(null);
    setIsScanning(true);

    try {
      const res = await fetch('/api/scan', { method: 'POST' });
      if (res.ok) {
        const data: ScanResult = await res.json();
        setResultOverlay(data);

        if (data.accepted) {
          confetti({
            particleCount: 100,
            spread: 75,
            origin: { y: 0.55 },
            colors: ['#FFDE59', '#38BDF8', '#FF4081', '#CCFF00', '#121212']
          });
        }

        // Auto-return to Idle state after 3.8s
        clearResultAutoReturn();
        resultAutoReturnTimerRef.current = setTimeout(() => {
          closeResultOverlay('idle');
        }, 3800);

      } else {
        const err = await res.json().catch(() => ({}));
        const rawDetail = err.detail || err.error_code || 'Palm not recognized';
        let friendlyMsg = rawDetail;
        if (rawDetail.includes('farther') || rawDetail.includes('too close') || rawDetail === 'HAND_TOO_CLOSE') {
          friendlyMsg = '⚠️ Hand too close — move hand farther (~10-15cm) from lens.';
        } else if (rawDetail.includes('closer') || rawDetail.includes('too far') || rawDetail === 'HAND_TOO_FAR') {
          friendlyMsg = '⚠️ Hand too far — move hand closer to sensor (~10-15cm).';
        } else if (rawDetail.includes('center') || rawDetail.includes('outside') || rawDetail === 'HAND_OUTSIDE_FRAME') {
          friendlyMsg = '⚠️ Place palm flat directly above the sensor.';
        } else if (rawDetail.includes('valleys') || rawDetail.includes('fingers') || rawDetail === 'VALLEY_EXTRACTION_FAILED') {
          friendlyMsg = '⚠️ Spread fingers slightly to expose knuckle valleys.';
        } else if (rawDetail.includes('QUALITY_LOW_CONTRAST')) {
          friendlyMsg = '⚠️ Low vein contrast — hold palm steady under sensor.';
        } else if (rawDetail.includes('QUALITY_EXCESSIVE_PADDING')) {
          friendlyMsg = '⚠️ Hand partially out of frame — center palm over lens.';
        }
        showToast(friendlyMsg, 'warn');
        setResultOverlay({
          accepted: false,
          username: null,
          score: 0.0,
          threshold: 0.2226,
          time_ms: 50
        });
        clearResultAutoReturn();
        resultAutoReturnTimerRef.current = setTimeout(() => {
          closeResultOverlay('scan');
        }, 3000);
      }
    } catch {
      showToast('Connection error. Please retry.', 'error');
    } finally {
      setIsScanning(false);
    }
  };

  // Live Camera Sample Capture with Dynamic Instruction Updates
  const handleCaptureSampleWithCountdown = async () => {
    if (isCapturingSample || enrollCountdown !== null || enrollSamples.length >= 6 || !cameraReady) return;
    const cleanUname = enrollUsername.trim().toLowerCase();
    if (!cleanUname) {
      showToast('Enter a username or identifier first!', 'warn');
      return;
    }

    const currentHint = ENROLL_SAMPLE_INSTRUCTIONS[enrollSamples.length] || 'Hold palm steady ~10-15cm above sensor';
    setEnrollStatusMsg(`${currentHint} (Capturing in 3 seconds...)`);

    // 3-Second Countdown
    for (let i = 3; i > 0; i--) {
      if (appStateRef.current !== 'enroll') {
        setEnrollCountdown(null);
        return;
      }
      setEnrollCountdown(i);
      await new Promise(r => setTimeout(r, 900));
    }
    if (appStateRef.current !== 'enroll') {
      setEnrollCountdown(null);
      return;
    }
    setEnrollCountdown(null);
    setIsCapturingSample(true);
    setEnrollStatusMsg('Acquiring frame & validating palm...');

    try {
      const res = await fetch('/api/enroll/sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: cleanUname, sample_idx: enrollSamples.length }),
      });
      
      if (res.ok) {
        // SUCCESS CASE: Read JSON once
        const data = await res.json();
        const nextSamples = [...enrollSamples, { vr_mean: data.vr_mean || 0.5, thumb: data.thumb || '' }];
        setEnrollSamples(nextSamples);
        const nextIdx = nextSamples.length;
        
        // Dynamically advance to the NEXT sample's instruction!
        if (nextIdx < 6) {
          const nextInstruction = ENROLL_SAMPLE_INSTRUCTIONS[nextIdx];
          setEnrollStatusMsg(`Sample #${nextIdx} saved! Next: ${nextInstruction}`);
        } else {
          setEnrollStatusMsg('All 6 samples captured! Click Save Enrollment below.');
        }
        showToast(`Sample ${nextIdx}/6 captured successfully!`, 'success');
      } else {
        // ERROR CASE: Read failure body once
        const err = await res.json().catch(() => ({}));
        const rawDetail = err.detail || err.error_code || 'Hand not detected';
        let friendlyMsg = rawDetail;
        if (rawDetail.includes('farther') || rawDetail.includes('too close') || rawDetail === 'HAND_TOO_CLOSE') {
          friendlyMsg = 'Hand too close — move hand farther (~10-15cm) from lens.';
        } else if (rawDetail.includes('closer') || rawDetail.includes('too far') || rawDetail === 'HAND_TOO_FAR') {
          friendlyMsg = 'Hand too far — move hand closer to sensor (~10-15cm).';
        } else if (rawDetail.includes('center') || rawDetail.includes('outside') || rawDetail === 'HAND_OUTSIDE_FRAME') {
          friendlyMsg = 'Center palm directly within the guide outline.';
        } else if (rawDetail.includes('valleys') || rawDetail.includes('fingers') || rawDetail === 'VALLEY_EXTRACTION_FAILED') {
          friendlyMsg = 'Spread fingers slightly and keep hand flat.';
        } else if (rawDetail.includes('QUALITY_LOW_CONTRAST')) {
          friendlyMsg = 'Low vein contrast — hold palm steady under sensor.';
        } else if (rawDetail.includes('QUALITY_EXCESSIVE_PADDING')) {
          friendlyMsg = 'Hand partially out of view — center palm.';
        }
        setEnrollStatusMsg(`Sample rejected: ${friendlyMsg}`);
        showToast(friendlyMsg, 'warn');
      }
    } catch {
      showToast('Terminal connection error.', 'error');
      setEnrollStatusMsg('Error communicating with terminal backend.');
    } finally {
      setIsCapturingSample(false);
    }
  };

  // Commit Multi-Sample Enrollment to Database
  const handleSaveEnrollment = async () => {
    const cleanUname = enrollUsername.trim().toLowerCase();
    if (enrollSamples.length < 3 || !cleanUname) {
      showToast('Capture at least 3 samples before saving!', 'warn');
      return;
    }
    try {
      const res = await fetch('/api/enroll/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: cleanUname }),
      });
      if (res.ok) {
        showToast(`Enrolled '${cleanUname}' successfully!`, 'success');
        confetti({
          particleCount: 120,
          spread: 85,
          origin: { y: 0.5 },
          colors: ['#FFDE59', '#38BDF8', '#FF4081', '#CCFF00']
        });
        setEnrollUsername('');
        setEnrollSamples([]);
        setEnrollStatusMsg('');
        loadStatus();
        setTimeout(() => setAppState('idle'), 1500);
      } else {
        const err = await res.json().catch(() => ({}));
        showToast(err.detail || 'Failed to save enrollment to database.', 'error');
      }
    } catch {
      showToast('Error saving template.', 'error');
    }
  };

  // Hidden Admin: Tap camera bead 5 times
  const handleSecretAdminTap = () => {
    const next = adminTapCount + 1;
    if (next >= 5) {
      setAdminTapCount(0);
      setAdminOpen(true);
      fetchReport();
    } else {
      setAdminTapCount(next);
      setTimeout(() => setAdminTapCount(0), 3000);
    }
  };

  const fetchReport = async () => {
    setReportLoading(true);
    try {
      const res = await fetch('/api/report');
      if (res.ok) {
        const data = await res.json();
        setReportData(data);
      }
    } catch {
      setReportData(null);
    } finally {
      setReportLoading(false);
    }
  };

  const handleDeleteUser = async (uname: string) => {
    if (!window.confirm(`Delete user '${uname}' and remove all templates?`)) return;
    try {
      const res = await fetch(`/api/users/${uname}`, { method: 'DELETE' });
      if (res.ok) {
        showToast(`User '${uname}' removed.`, 'success');
        fetchReport();
        loadStatus();
      }
    } catch {
      showToast('Error removing user.', 'error');
    }
  };

  const [isCleaningDb, setIsCleaningDb] = useState(false);

  const handleCleanDatabase = async () => {
    if (!window.confirm("⚠️ CLEAN USERS DATABASE FOR DEMO?\n\nThis will remove all enrolled users and biometric templates so you can perform a clean, error-free live demonstration.")) {
      return;
    }
    setIsCleaningDb(true);
    try {
      const res = await fetch('/api/database/reset', { method: 'POST' });
      if (res.ok) {
        showToast("Database cleaned! 0 users enrolled — ready for fresh demo.", "success");
        fetchReport();
        loadStatus();
      } else {
        const err = await res.json().catch(() => ({}));
        showToast(err.detail || "Failed to reset database.", "error");
      }
    } catch {
      showToast("Error connecting to server.", "error");
    } finally {
      setIsCleaningDb(false);
    }
  };

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-dribbble-yellow flex justify-center items-center p-0 sm:p-4 text-[#121212] select-none font-sans">
       
        {/* ── 5" RASPBERRY PI TOUCH DISPLAY FRAME ── */}
        <div className="w-full max-w-[760px] h-[98vh] max-h-[1200px] bg-[#FFFDF0] border-[4px] border-black rounded-[24px] shadow-[8px_8px_0px_#121212] flex flex-col relative overflow-hidden bg-neo-cream">

          {/* ── HEADER: STATUS (LIVE) + TITLE (PALM VEIN BIOMETRICS) + CLOCK ── */}
          <div className="px-5 pt-3 pb-2.5 flex items-center justify-between text-xs font-black text-black z-20 border-b-[2px] border-black/10">
            <div className="flex items-center gap-2">
              {/* Secret 5-tap Admin trigger on camera status bead */}
              <button 
                onClick={handleSecretAdminTap}
                className="flex items-center gap-1.5 focus:outline-none cursor-pointer"
                title="Status indicator"
              >
                <span className={`w-3 h-3 rounded-full border-[1.5px] border-black transition-colors ${
                  cameraState === 'live' ? 'bg-[#CCFF00]' : cameraState === 'connecting' ? 'bg-[#FFDE59] animate-ping' : 'bg-[#FF4081]'
                }`} />
                {/* Shows "LIVE" in left corner as requested */}
                <span className="text-[11px] uppercase font-bold tracking-tight text-[#222]">
                  {cameraState === 'live' ? 'LIVE' : cameraState === 'connecting' ? 'CONNECTING' : 'OFFLINE'}
                </span>
              </button>
            </div>

            <div className="text-center font-display font-black text-xs tracking-wider uppercase text-black">
              PALM VEIN BIOMETRICS
            </div>

            <div className="text-[11px] font-mono font-black text-[#555]">
              {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </div>
          </div>

          {/* ── TOAST ALERT BANNER ── */}
          {toast && (
            <div className="absolute top-14 left-5 right-5 z-50 animate-bounce">
              <div className={`p-3 border-[3px] border-black rounded-2xl shadow-[4px_4px_0px_#121212] font-display font-black text-xs text-center flex items-center justify-center gap-2 ${
                toast.type === 'error' ? 'bg-[#FF4081] text-white' : toast.type === 'warn' ? 'bg-[#FF7A00] text-white' : 'bg-[#CCFF00] text-black'
              }`}>
                {toast.type === 'error' || toast.type === 'warn' ? <AlertTriangle className="w-4 h-4" /> : <CheckCircle2 className="w-4 h-4" />}
                <span>{toast.msg}</span>
              </div>
            </div>
          )}

          {/* ══════════════════════════════════════════════════════════════════════
              SCREEN 1: IDLE / HOME SCREEN
              - Soft, minimal, clean aesthetic
              - NO instructions on the home page (per user requirement)
              - Accommodates BOTH palms
              - Two clear, big action buttons with text inside
             ══════════════════════════════════════════════════════════════════════ */}
          {appState === 'idle' && (
            <div className="flex-1 flex flex-col items-center justify-between p-6 animate-fadeIn relative overflow-y-auto select-none">
              
              {/* Minimal Top Status Pill */}
              <div className="w-full flex justify-center items-center z-10 pt-1">
                <div className="px-4 py-1 bg-white border-[2px] border-black rounded-full text-[11px] font-black shadow-[2px_2px_0px_#121212] flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-[#CCFF00] animate-pulse" />
                  <span>TERMINAL READY</span>
                </div>
              </div>

              {/* Center Card: Soft, Simple, Minimal */}
              <div className="w-full max-w-[530px] bg-[#FFFDF0] border-[4px] border-black rounded-3xl p-6 sm:p-7 shadow-[8px_8px_0px_#121212] flex flex-col items-center text-center my-auto z-10">
               
                {/* Soft Minimal Biometric Palm Icon */}
                <div className="relative my-2 w-20 h-20 rounded-3xl bg-white border-[3.5px] border-black shadow-[4px_4px_0px_#121212] flex items-center justify-center">
                  <Scan className="w-14 h-14 text-[#FFDE59] stroke-[2.5]" />
                  <Hand className="w-8 h-8 text-black stroke-[2] absolute" />
                </div>

                <div className="space-y-2 mt-3">
                  <h1 className="font-display font-black text-2xl sm:text-3xl leading-tight uppercase tracking-tight">
                    Present your palm<br />to begin
                  </h1>
                  <p className="text-xs sm:text-sm font-bold text-[#555] max-w-[340px] mx-auto leading-snug">
                    Hold your hand flat, 40 cm above the sensor. No contact needed
                  </p>
                </div>

                {/* ── SIDE-BY-SIDE RECTANGULAR ACTION BUTTONS ── */}
                <div className="w-full grid grid-cols-2 gap-3 mt-6">
                  
                  {/* BUTTON 1: SCAN PALM (SIDE-BY-SIDE RECTANGULAR BUTTON) */}
                  <button
                    onClick={() => setAppState('scan')}
                    className="py-4 px-3.5 bg-[#FFDE59] hover:bg-[#ffe680] text-black border-[3.5px] border-black rounded-2xl shadow-[4px_4px_0px_#121212] font-display font-black flex items-center justify-center gap-3 neo-btn cursor-pointer"
                  >
                    <div className="w-10 h-10 rounded-xl bg-black text-[#FFDE59] flex items-center justify-center border-[2px] border-black shadow-[2px_2px_0px_#121212] shrink-0">
                      <Scan className="w-5 h-5 stroke-[2.5]" />
                    </div>
                    <div className="text-left">
                      <div className="text-sm sm:text-base font-black tracking-tight uppercase leading-none">
                        SCAN PALM
                      </div>
                      <div className="text-[10px] font-bold text-[#555] normal-case leading-tight mt-0.5">
                        Verify identity
                      </div>
                    </div>
                  </button>

                  {/* BUTTON 2: ENROLL PALM (SIDE-BY-SIDE RECTANGULAR BUTTON) */}
                  <button
                    onClick={() => setAppState('enroll')}
                    className="py-4 px-3.5 bg-[#CCFF00] hover:bg-[#d9ff33] text-black border-[3.5px] border-black rounded-2xl shadow-[4px_4px_0px_#121212] font-display font-black flex items-center justify-center gap-3 neo-btn cursor-pointer"
                  >
                    <div className="w-10 h-10 rounded-xl bg-black text-[#CCFF00] flex items-center justify-center border-[2px] border-black shadow-[2px_2px_0px_#121212] shrink-0">
                      <UserPlus className="w-5 h-5 stroke-[2.5]" />
                    </div>
                    <div className="text-left">
                      <div className="text-sm sm:text-base font-black tracking-tight uppercase leading-none">
                        ENROLL PALM
                      </div>
                      <div className="text-[10px] font-bold text-[#555] normal-case leading-tight mt-0.5">
                        Register user
                      </div>
                    </div>
                  </button>

                </div>
              </div>

              {/* Interactive Bottom Feature Badges - FLOATING ANIMATION */}
              <div className="w-full text-center z-10 pt-4 pb-2">
                <div className="text-[10px] font-black uppercase tracking-[0.2em] text-[#888] mb-3">
                  POWERED BY SUB-DERMAL VASCULAR INTELLIGENCE
                </div>

                {/* Floating keyword pills + Clean Users Demo Button — scattered at big scale */}
                <div className="relative h-44 sm:h-48 w-full max-w-[650px] mx-auto select-none">

                  {/* VEIN MAPPING — top-left, floats slow */}
                  <span
                    className="absolute px-4 py-2 bg-[#FFDE59] text-black border-[2.5px] border-black rounded-2xl font-display font-black text-xs shadow-[3px_3px_0px_#121212] select-none pointer-events-none"
                    style={{
                      animation: 'floatBadge 3.6s ease-in-out infinite',
                      left: '2%',
                      top: '6px',
                    }}
                  >
                    🩸 VEIN MAPPING
                  </span>

                  {/* NIR SENSOR — top-right, floats medium */}
                  <span
                    className="absolute px-4 py-2 bg-[#38BDF8] text-black border-[2.5px] border-black rounded-2xl font-display font-black text-xs shadow-[3px_3px_0px_#121212] select-none pointer-events-none"
                    style={{
                      animation: 'floatBadge 4.2s ease-in-out infinite',
                      animationDelay: '0.9s',
                      right: '2%',
                      top: '12px',
                    }}
                  >
                    📡 NIR SENSOR
                  </span>

                  {/* CLEAN USERS DATA (DEMO RESET BUTTON) — centered & floating */}
                  <div className="absolute left-1/2 -translate-x-1/2 top-[48px] z-20">
                    <button
                      onClick={handleCleanDatabase}
                      disabled={isCleaningDb}
                      title="Clean all enrolled users from database for a fresh demo"
                      className="px-4 py-2 bg-white text-black border-[2.5px] border-black rounded-2xl font-display font-black text-xs shadow-[3px_3px_0px_#121212] neo-btn hover:bg-[#FFE5E5] active:translate-x-[2px] active:translate-y-[2px] cursor-pointer flex items-center gap-1.5 select-none"
                      style={{
                        animation: 'floatBadge 4.6s ease-in-out infinite',
                        animationDelay: '1.2s',
                      }}
                    >
                      {isCleaningDb ? (
                        <RefreshCw className="w-3.5 h-3.5 animate-spin text-[#FF4081]" />
                      ) : (
                        <Trash2 className="w-3.5 h-3.5 text-[#FF4081] stroke-[2.5]" />
                      )}
                      <span>CLEAN USERS DATA</span>
                      {totalUsers > 0 && (
                        <span className="px-1.5 py-0.5 bg-[#FF4081] text-white rounded-full text-[10px] font-black leading-tight">
                          {totalUsers}
                        </span>
                      )}
                    </button>
                  </div>

                  {/* ZERO CONTACT — bottom-left, floats fast */}
                  <span
                    className="absolute px-4 py-2 bg-[#CCFF00] text-black border-[2.5px] border-black rounded-2xl font-display font-black text-xs shadow-[3px_3px_0px_#121212] select-none pointer-events-none"
                    style={{
                      animation: 'floatBadge 5.0s ease-in-out infinite',
                      animationDelay: '1.7s',
                      left: '10%',
                      top: '96px',
                    }}
                  >
                    ✋ ZERO CONTACT
                  </span>

                  {/* LIVE TISSUE — bottom-right, floats slowest */}
                  <span
                    className="absolute px-4 py-2 bg-[#FF4081] text-white border-[2.5px] border-black rounded-2xl font-display font-black text-xs shadow-[3px_3px_0px_#121212] select-none pointer-events-none"
                    style={{
                      animation: 'floatBadge 3.0s ease-in-out infinite',
                      animationDelay: '2.5s',
                      right: '10%',
                      top: '90px',
                    }}
                  >
                    💡 LIVE TISSUE
                  </span>

                </div>
              </div>
            </div>
          )}

          {/* ══════════════════════════════════════════════════════════════════════
              SCREEN 2: SCAN (CLEAR CAMERA FEED WITH EQUAL VIEWPORT)
              - Unobstructed camera view (same size as Enroll camera)
              - Soft minimal translucent guide (fits either left or right palm)
              - Clean 3-second countdown
             ══════════════════════════════════════════════════════════════════════ */}
          {appState === 'scan' && (
            <div className="flex-1 flex flex-col p-4 space-y-3 animate-fadeIn overflow-y-auto pb-4">
             
              {/* Header Bar: Centered Mode Badge (Duplicate top buttons removed) */}
              <div className="flex items-center justify-center">
                <div className="px-4 py-1.5 bg-[#FFDE59] border-[2px] border-black rounded-xl text-xs font-black shadow-[2px_2px_0px_#121212] flex items-center gap-1.5">
                  <ShieldCheck className="w-3.5 h-3.5 stroke-[2.5]" />
                  <span>PALM SCAN MODE</span>
                </div>
              </div>

              {/* Dominant Camera Viewport — enlarged to fill the kiosk screen */}
              <div className="w-full relative flex flex-col items-center shrink-0">
                <CameraViewport 
                  cameraState={cameraState}
                  cameraErrorDetail={cameraErrorDetail}
                  onRetry={loadStatus}
                  className="w-full h-[580px] sm:h-[640px]"
                >
                  {/* Soft Minimal Guide - Fits Either Hand, Unobstructed Video */}
                  <SoftPalmGuide />

                  {/* 3-Second Countdown Overlay */}
                  {scanCountdown !== null && (
                    <div className="absolute inset-0 bg-black/50 flex flex-col items-center justify-center animate-fadeIn z-30">
                      <span className="font-display font-black text-8xl text-[#FFDE59] drop-shadow-[4px_4px_0px_#000] animate-bounce">
                        {scanCountdown}
                      </span>
                    </div>
                  )}

                  {/* Processing Spinner Overlay */}
                  {isScanning && (
                    <div className="absolute inset-0 bg-black/70 flex flex-col items-center justify-center animate-fadeIn text-white z-30 space-y-2">
                      <RefreshCw className="w-10 h-10 animate-spin text-[#38BDF8]" />
                      <span className="text-sm font-black text-[#CCFF00] tracking-wider uppercase">
                        VERIFYING PALM...
                      </span>
                    </div>
                  )}
                </CameraViewport>
              </div>

              {/* Status Guidance Indicator (Below the camera, not covering the video) */}
              <div className="text-center">
                <span className={`inline-block px-4 py-1.5 rounded-full text-xs font-black border-[2px] border-black shadow-[2px_2px_0px_#121212] ${
                  scanCountdown !== null 
                    ? 'bg-[#FFDE59] text-black animate-pulse' 
                    : isScanning 
                    ? 'bg-[#38BDF8] text-black' 
                    : cameraState === 'live' 
                    ? 'bg-[#CCFF00] text-black' 
                    : 'bg-[#FF4081] text-white'
                }`}>
                  {scanCountdown !== null 
                    ? `HOLD STEADY: SCANNING IN ${scanCountdown}S` 
                    : isScanning 
                    ? 'PROCESSING PALM VEIN...' 
                    : cameraState === 'live'
                    ? 'READY TO SCAN'
                    : 'CAMERA OFFLINE'}
                </span>
              </div>

              {/* Big Scan Button */}
              <button
                onClick={handleScanWithCountdown}
                disabled={isScanning || scanCountdown !== null || cameraState !== 'live' || !modelLoaded}
                className="w-full py-4 bg-[#FFDE59] text-black border-[3.5px] border-black rounded-2xl shadow-[5px_5px_0px_#121212] font-display font-black text-lg flex items-center justify-center gap-3 neo-btn hover:bg-[#ffe26b] disabled:bg-[#E2E8F0] disabled:text-[#888888] disabled:border-[#888888] disabled:shadow-none disabled:cursor-not-allowed cursor-pointer"
              >
                {scanCountdown !== null ? (
                  <>
                    <Timer className="w-6 h-6 animate-spin" />
                    <span>HOLD STEADY: {scanCountdown}s...</span>
                  </>
                ) : isScanning ? (
                  <>
                    <RefreshCw className="w-6 h-6 animate-spin" />
                    <span>READING SENSOR...</span>
                  </>
                ) : cameraState !== 'live' ? (
                  <>
                    <AlertTriangle className="w-5 h-5 text-[#888]" />
                    <span>CAMERA OFFLINE</span>
                  </>
                ) : !modelLoaded ? (
                  <>
                    <AlertTriangle className="w-5 h-5 text-[#888]" />
                    <span>MODEL NOT LOADED</span>
                  </>
                ) : (
                  <>
                    <Eye className="w-6 h-6 stroke-[3]" />
                    <span>SCAN PALM NOW</span>
                  </>
                )}
              </button>

              {/* Secondary Action Row: Cancel Scan & Enroll New Palm */}
              <div className="grid grid-cols-2 gap-3 pt-1">
                <button
                  onClick={() => setAppState('idle')}
                  className="py-3 px-4 bg-white text-black border-[3px] border-black rounded-xl shadow-[3px_3px_0px_#121212] font-display font-black text-xs neo-btn hover:bg-[#f5f5f0] flex items-center justify-center gap-1.5 cursor-pointer"
                >
                  <ArrowLeft className="w-4 h-4 stroke-[2.5]" />
                  <span>CANCEL SCAN</span>
                </button>

                <button
                  onClick={() => setAppState('enroll')}
                  className="py-3 px-4 bg-[#CCFF00] text-black border-[3px] border-black rounded-xl shadow-[3px_3px_0px_#121212] font-display font-black text-xs neo-btn hover:bg-[#b8e600] flex items-center justify-center gap-1.5 cursor-pointer"
                >
                  <UserPlus className="w-4 h-4 stroke-[2.5]" />
                  <span>ENROLL PALM</span>
                </button>
              </div>
            </div>
          )}

          {/* ══════════════════════════════════════════════════════════════════════
              SCREEN 3: ENROLLMENT (6 GUIDED SAMPLES WITH DYNAMIC INSTRUCTIONS)
              - Generous camera viewport (nearly same size as scan screen)
              - Whole palm visible with object-contain (zero clipping)
              - Dynamic instruction banner directly below camera
              - Ergonomic Admin control row (Username input + Capture button side by side)
              - 6-Cell calibration progress matrix
              - Prominent Save Enrollment button when >=3 samples ready
             ══════════════════════════════════════════════════════════════════════ */}
          {appState === 'enroll' && (
            <div className="flex-1 flex flex-col p-4 space-y-3 animate-fadeIn overflow-y-auto pb-4">
             
              {/* Header: Title + Mode Toggle Buttons */}
              <div className="flex items-center justify-between border-b-[2px] border-black/10 pb-2">
                <button
                  onClick={() => setAppState('idle')}
                  className="px-3 py-1.5 bg-white border-[2px] border-black rounded-xl text-xs font-black shadow-[2px_2px_0px_#121212] flex items-center gap-1.5 neo-btn cursor-pointer"
                >
                  <ArrowLeft className="w-3.5 h-3.5 stroke-[3]" />
                  <span>Cancel</span>
                </button>

                <div className="text-center">
                  <h2 className="font-display font-black text-base tracking-tight uppercase">ENROLL PALM</h2>
                  <p className="text-[10px] font-bold text-[#666]">6 guided biometric samples</p>
                </div>

                <button
                  onClick={() => setAppState('scan')}
                  className="px-3 py-1.5 bg-[#FFDE59] border-[2px] border-black rounded-xl text-xs font-black shadow-[2px_2px_0px_#121212] flex items-center gap-1.5 neo-btn cursor-pointer"
                  title="Switch to Scan Mode"
                >
                  <Scan className="w-3.5 h-3.5" />
                  <span>Scan</span>
                </button>
              </div>

              {/* Camera Viewport — SAME LARGE SIZE AS SCAN SECTION */}
              <div className="w-full relative flex flex-col items-center shrink-0">
                <CameraViewport
                  cameraState={cameraState}
                  cameraErrorDetail={cameraErrorDetail}
                  onRetry={loadStatus}
                  className="w-full h-[580px] sm:h-[640px]"
                >
                  {/* Soft Minimal Guide - Fits Either Hand */}
                  <SoftPalmGuide />

                  {/* Clean Countdown Overlay */}
                  {enrollCountdown !== null && (
                    <div className="absolute inset-0 bg-black/50 flex flex-col items-center justify-center z-20">
                      <span className="font-display font-black text-7xl text-[#FFDE59] drop-shadow-[3px_3px_0px_#000] animate-bounce">
                        {enrollCountdown}
                      </span>
                    </div>
                  )}

                  {isCapturingSample && (
                    <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center text-white z-20 space-y-1">
                      <RefreshCw className="w-8 h-8 animate-spin text-[#38BDF8]" />
                      <span className="text-xs font-black text-[#CCFF00] uppercase tracking-wider">
                        PROCESSING SAMPLE...
                      </span>
                    </div>
                  )}
                </CameraViewport>
              </div>

              {/* ── DYNAMIC INSTRUCTION BANNER (DIRECTLY BELOW CAMERA, CHANGES AFTER EVERY SAMPLE) ── */}
              <div className="border-[2px] border-black rounded-xl p-2.5 shadow-[2px_2px_0px_#121212] bg-[#FFDE59] text-black">
                <div className="flex items-center gap-2">
                  <Info className="w-5 h-5 shrink-0" />
                  <span className="text-xs font-black leading-tight">
                    {enrollStatusMsg || ENROLL_SAMPLE_INSTRUCTIONS[enrollSamples.length] || "Calibration complete. Click Save below."}
                  </span>
                </div>
              </div>

              {/* ── ADMIN WORKFLOW CARD: USERNAME INPUT + CAPTURE BUTTON SIDE BY SIDE ── */}
              <div className="bg-white border-[3px] border-black rounded-2xl p-3 shadow-[3px_3px_0px_#121212] space-y-3">
                <div className="flex items-end gap-2.5">
                  <div className="flex-1 space-y-1">
                    <label className="text-[10px] font-black uppercase tracking-wider text-black">
                      USERNAME / IDENTIFIER
                    </label>
                    <input
                      type="text"
                      value={enrollUsername}
                      onChange={e => setEnrollUsername(e.target.value.toLowerCase().replace(/[^a-z0-9_-]/g, ''))}
                      placeholder="e.g. user_alpha"
                      className="w-full px-3 py-2 bg-white border-[2.5px] border-black rounded-xl shadow-[2px_2px_0px_#121212] font-display font-black text-sm outline-none focus:bg-[#FFFDF0]"
                    />
                  </div>

                  <button
                    onClick={handleCaptureSampleWithCountdown}
                    disabled={isCapturingSample || enrollCountdown !== null || enrollSamples.length >= 6 || !enrollUsername.trim() || cameraState !== 'live' || !modelLoaded}
                    className={`h-[40px] px-4 border-[2.5px] border-black rounded-xl shadow-[2px_2px_0px_#121212] font-display font-black text-xs flex items-center justify-center gap-2 neo-btn disabled:bg-[#E2E8F0] disabled:text-[#888888] disabled:border-[#888888] disabled:shadow-none disabled:cursor-not-allowed cursor-pointer whitespace-nowrap shrink-0 ${
                      enrollCountdown !== null ? 'bg-[#FFDE59] text-black animate-pulse' : 'bg-[#38BDF8] text-black hover:bg-[#2cb0eb]'
                    }`}
                  >
                    {enrollCountdown !== null ? (
                      <>
                        <Timer className="w-4 h-4 animate-spin" />
                        <span>CAPTURING: {enrollCountdown}s</span>
                      </>
                    ) : isCapturingSample ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        <span>PROCESSING...</span>
                      </>
                    ) : (
                      <>
                        <Camera className="w-4 h-4 stroke-[2.5]" />
                        <span>CAPTURE #{enrollSamples.length + 1}</span>
                      </>
                    )}
                  </button>
                </div>

                {/* 6-Cell Sample Matrix Grid */}
                <div className="space-y-1.5 pt-1 border-t-[1.5px] border-black/10">
                  <div className="flex justify-between items-center text-[11px] font-black">
                    <span className="uppercase tracking-wider">CALIBRATION PROGRESS:</span>
                    <span className="px-2 py-0.5 bg-[#38BDF8] border-[1.5px] border-black rounded-full text-[10px]">
                      {enrollSamples.length} / 6 SAMPLES {enrollSamples.length >= 3 ? '(READY TO SAVE)' : ''}
                    </span>
                  </div>

                  <div className="grid grid-cols-6 gap-1.5">
                    {[0, 1, 2, 3, 4, 5].map(idx => {
                      const sample = enrollSamples[idx];
                      const isDone = !!sample;
                      return (
                        <div
                          key={idx}
                          className={`h-10 rounded-xl border-[2px] border-black shadow-[2px_2px_0px_#121212] flex items-center justify-center font-display font-black text-xs transition-all overflow-hidden ${
                            isDone ? 'bg-[#CCFF00] scale-105' : idx < 3 ? 'bg-[#FFFDE8] text-[#888]' : 'bg-[#F4F4F0] text-[#aaa]'
                          }`}>
                          {isDone && sample.thumb ? (
                            <img src={`data:image/png;base64,${sample.thumb}`} alt={`Sample ${idx+1}`} className="w-full h-full object-cover" />
                          ) : isDone ? (
                            '✓'
                          ) : (
                            `#${idx + 1}`
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Save & Commit Button */}
                {enrollSamples.length >= 3 && (
                  <button
                    onClick={handleSaveEnrollment}
                    className="w-full py-3 bg-[#CCFF00] text-black border-[3px] border-black rounded-xl shadow-[3px_3px_0px_#121212] font-display font-black text-sm flex items-center justify-center gap-2 neo-btn hover:bg-[#b8e600] animate-bounce cursor-pointer"
                  >
                    <Check className="w-5 h-5 stroke-[3]" />
                    <span>SAVE ENROLLMENT ({enrollSamples.length} SAMPLES)</span>
                  </button>
                )}
              </div>
            </div>
          )}

          {/* ══════════════════════════════════════════════════════════════════════
              FULLSCREEN RESULT OVERLAY (SUCCESS / FAILURE MOMENT)
              - Clean match / rejection feedback
              - Auto-returns after 3.8s
             ══════════════════════════════════════════════════════════════════════ */}
          {resultOverlay && (
            <div className={`absolute inset-0 z-50 p-6 flex flex-col items-center justify-center animate-fadeIn ${
              resultOverlay.accepted ? 'bg-[#CCFF00]' : 'bg-[#FF4081]'
            }`}>
              <div className="w-full max-w-[460px] bg-[#FFFDF0] border-[4px] border-black rounded-3xl p-6 shadow-[8px_8px_0px_#121212] text-center space-y-4 neo-card">
               
                {/* Status Icon */}
                <div className="relative mx-auto w-20 h-20">
                  <div className={`w-20 h-20 rounded-2xl border-[3.5px] border-black shadow-[4px_4px_0px_#121212] flex items-center justify-center font-display font-black text-4xl animate-float ${
                    resultOverlay.accepted ? 'bg-[#CCFF00] text-black' : 'bg-[#FF4081] text-white'
                  }`}>
                    {resultOverlay.accepted ? '✓' : '✕'}
                  </div>
                </div>

                {/* Headline & Identity */}
                <div className="space-y-1">
                  <h3 className="font-display font-black text-2xl tracking-tight uppercase leading-tight">
                    {resultOverlay.accepted ? 'PALM VERIFIED' : 'NOT RECOGNIZED'}
                  </h3>
                  <p className="font-bold text-xs text-[#555]">
                    {resultOverlay.accepted 
                      ? `Welcome, ${resultOverlay.username?.toUpperCase()}! Identity confirmed.` 
                      : 'Vein pattern not recognized. Reposition palm and try again.'}
                  </p>
                </div>

                {/* Telemetry Card */}
                <div className="bg-white border-[2px] border-black rounded-2xl p-3 shadow-[2px_2px_0px_#121212] space-y-1.5 text-left">
                  <div className="flex justify-between items-center text-xs font-black">
                    <span className="text-[#666] uppercase">RESULT:</span>
                    <span className={`px-2 py-0.5 rounded border-[1.5px] border-black text-[10px] ${
                      resultOverlay.accepted ? 'bg-[#CCFF00]' : 'bg-[#FF4081] text-white'
                    }`}>
                      {resultOverlay.accepted ? 'MATCH CONFIRMED' : 'REJECTED'}
                    </span>
                  </div>

                  <div className="flex justify-between items-center text-[10px] font-mono font-bold text-[#666]">
                    <span>Similarity: {resultOverlay.score.toFixed(4)} (Threshold: {resultOverlay.threshold.toFixed(4)})</span>
                    <span>Latency: {resultOverlay.time_ms} ms</span>
                  </div>

                  {resultOverlay.clahe_base64 && (
                    <div className="pt-2 border-t-[1.5px] border-black/10 flex items-center justify-between">
                      <span className="text-[10px] font-black text-[#555] uppercase">EXTRACTED VEIN ROI:</span>
                      <img 
                        src={`data:image/png;base64,${resultOverlay.clahe_base64}`} 
                        alt="Enhanced ROI" 
                        className="w-8 h-8 rounded-lg border-[1.5px] border-black object-cover bg-black"
                      />
                    </div>
                  )}
                </div>

                {/* Action Buttons */}
                <div className="grid grid-cols-2 gap-3 pt-1">
                  <button
                    onClick={() => closeResultOverlay('scan')}
                    className="py-3 bg-[#CCFF00] text-black border-[3px] border-black rounded-xl shadow-[3px_3px_0px_#121212] font-display font-black text-xs neo-btn hover:bg-[#b8e600] flex items-center justify-center gap-1.5 cursor-pointer"
                  >
                    <RefreshCw className="w-3.5 h-3.5 stroke-[2.5]" />
                    <span>SCAN AGAIN</span>
                  </button>

                  <button
                    onClick={() => closeResultOverlay('idle')}
                    className="py-3 bg-[#FFDE59] text-black border-[3px] border-black rounded-xl shadow-[3px_3px_0px_#121212] font-display font-black text-xs neo-btn hover:bg-[#ffe26b] flex items-center justify-center gap-1.5 cursor-pointer"
                  >
                    <ArrowLeft className="w-3.5 h-3.5 stroke-[2.5]" />
                    <span>EXIT (3s)</span>
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* ══════════════════════════════════════════════════════════════════════
              HIDDEN OPERATOR / ADMIN MODAL (5 Taps on LIVE status bead)
             ══════════════════════════════════════════════════════════════════════ */}
          {adminOpen && (
            <div className="absolute inset-0 bg-black/80 z-50 flex items-center justify-center p-4 animate-fadeIn">
              <div className="w-full max-h-[92%] bg-[#FFFDF0] border-[4px] border-black rounded-3xl p-5 shadow-[8px_8px_0px_#121212] flex flex-col space-y-3 overflow-hidden text-xs">
               
                <div className="flex justify-between items-center border-b-2 border-black pb-2">
                  <div className="flex items-center gap-2">
                    <Lock className="w-4 h-4 text-black" />
                    <h3 className="font-display font-black text-sm uppercase">OPERATOR SETTINGS</h3>
                  </div>
                  <button 
                    onClick={() => setAdminOpen(false)} 
                    className="w-7 h-7 bg-[#F4F4F0] border-[1.5px] border-black rounded-lg flex items-center justify-center font-black cursor-pointer"
                  >
                    ✕
                  </button>
                </div>

                <div className="overflow-y-auto space-y-3 flex-1 pr-1">
                  {/* System & Hardware Specs */}
                  <div className="p-3 bg-white border-[2px] border-black rounded-xl space-y-1.5 shadow-[2px_2px_0px_#121212]">
                    <div className="flex items-center gap-1.5 text-[#38BDF8] font-black pb-1 border-b border-black/10">
                      <Cpu className="w-3.5 h-3.5" />
                      <span className="uppercase text-[11px]">HARDWARE & ENGINE</span>
                    </div>
                    <div className="flex justify-between font-black">
                      <span>ENGINE:</span>
                      <span className="font-mono">AMPVNet ONNX (v2)</span>
                    </div>
                    <div className="flex justify-between font-black">
                      <span>SENSOR STATUS:</span>
                      <span className="uppercase text-[#38BDF8] font-mono">ONLINE</span>
                    </div>
                    <div className="flex justify-between font-black">
                      <span>ENROLLED USERS:</span>
                      <span className="font-mono">{totalUsers} Users</span>
                    </div>
                    <div className="flex justify-between font-black">
                      <span>MATCH THRESHOLD:</span>
                      <span className="font-mono">0.45</span>
                    </div>
                  </div>

                  {/* Registered Users Management */}
                  <div className="p-3 bg-white border-[2px] border-black rounded-xl space-y-2 shadow-[2px_2px_0px_#121212]">
                    <div className="flex justify-between items-center">
                      <span className="font-display font-black text-xs uppercase">REGISTERED USERS</span>
                      <button 
                        onClick={fetchReport} 
                        disabled={reportLoading}
                        className="px-2 py-0.5 bg-[#38BDF8] border-[1.5px] border-black rounded text-[10px] font-black neo-btn cursor-pointer"
                      >
                        {reportLoading ? 'Loading...' : 'Refresh'}
                      </button>
                    </div>

                    {reportData?.users && reportData.users.length > 0 ? (
                      <div className="space-y-1 max-h-40 overflow-y-auto">
                        {reportData.users.map(u => (
                          <div key={u.username} className="p-1.5 bg-[#F8F8F4] border border-black/10 rounded flex justify-between items-center font-mono text-[10px]">
                            <span>{u.username} ({u.sample_count} samples)</span>
                            <button
                              onClick={() => handleDeleteUser(u.username)}
                              className="px-1.5 py-0.5 bg-[#FF4081] text-white border border-black rounded text-[9px] font-bold cursor-pointer"
                            >
                              Delete
                            </button>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-[10px] font-bold text-[#888]">No enrolled users in database.</p>
                    )}
                  </div>
                </div>

                <div className="pt-2 border-t-2 border-black flex justify-end">
                  <button
                    onClick={() => setAdminOpen(false)}
                    className="px-4 py-2 bg-[#FFDE59] border-[2px] border-black rounded-xl font-black text-xs shadow-[2px_2px_0px_#121212] neo-btn cursor-pointer"
                  >
                    Close
                  </button>
                </div>
              </div>
            </div>
          )}

        </div>
      </div>
    </ErrorBoundary>
  );
}