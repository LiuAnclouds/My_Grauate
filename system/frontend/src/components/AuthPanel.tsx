import { CSSProperties, FormEvent, PointerEvent, useEffect, useMemo, useState } from "react";
import { AuthResponse, fetchLoginCaptcha, LoginCaptchaResponse, login, register, requestCode } from "../services/api";

type Props = {
  onAuthed: (session: AuthResponse) => void;
};

type FocusField = "account" | "password" | "register-code" | "login-captcha" | null;
type MascotMode = "idle" | "peek" | "cover" | "error";
type MascotTarget = "idle" | "account" | "password" | "captcha" | "error";
type FeedbackType = "info" | "success" | "error";

const footerFacts = ["面向交易关系的风险识别", "辅助定位异常用户", "支持数据接入与图谱分析"];

function EyeIcon({ open }: { open: boolean }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      {open ? (
        <>
          <path d="M2.5 12s3.4-6 9.5-6 9.5 6 9.5 6-3.4 6-9.5 6-9.5-6-9.5-6Z" />
          <circle cx="12" cy="12" r="2.8" />
        </>
      ) : (
        <>
          <path d="M3 3l18 18" />
          <path d="M9.4 5.4A10.7 10.7 0 0 1 12 5c6.1 0 9.5 7 9.5 7a17.6 17.6 0 0 1-3.1 3.9" />
          <path d="M14.1 14.2A3 3 0 0 1 9.8 9.9" />
          <path d="M6.5 6.8C3.9 8.5 2.5 12 2.5 12s3.4 7 9.5 7c1.5 0 2.8-.4 4-1" />
        </>
      )}
    </svg>
  );
}

function RefreshIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
      <path d="M20 11a8.1 8.1 0 0 0-14.4-4.8L4 8" />
      <path d="M4 4v4h4" />
      <path d="M4 13a8.1 8.1 0 0 0 14.4 4.8L20 16" />
      <path d="M20 20v-4h-4" />
    </svg>
  );
}

function AuthMascot({
  side,
  mode,
  target,
  style
}: {
  side: "left" | "right";
  mode: MascotMode;
  target: MascotTarget;
  style: CSSProperties;
}) {
  return (
    <aside className={`mascot-stage mascot-${side} mascot-${mode} target-${target}`} style={style} aria-hidden="true">
      <svg className="mascot-svg" viewBox="0 0 220 280">
        <defs>
          <linearGradient id={`pandaBody-${side}`} x1="40" x2="178" y1="128" y2="238">
            <stop offset="0%" stopColor="#ffffff" />
            <stop offset="100%" stopColor="#d8eef9" />
          </linearGradient>
          <linearGradient id={`buddyBody-${side}`} x1="48" x2="180" y1="126" y2="238">
            <stop offset="0%" stopColor="#ffe27a" />
            <stop offset="54%" stopColor="#fbbf24" />
            <stop offset="100%" stopColor="#3dd6c6" />
          </linearGradient>
          <filter id={`mascotShadow-${side}`} x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="12" stdDeviation="10" floodColor="#244060" floodOpacity="0.18" />
          </filter>
        </defs>

        <ellipse className="mascot-ground" cx="110" cy="252" rx="62" ry="16" />
        <g className="mascot-sparkles">
          <circle cx="38" cy="62" r="5" />
          <circle cx="184" cy="80" r="4" />
          <path d="M176 42l5 10 10 5-10 5-5 10-5-10-10-5 10-5z" />
        </g>

        {side === "left" ? (
          <g className="mascot-character panda-character" filter={`url(#mascotShadow-${side})`}>
            <g className="mascot-body-svg">
              <path className="panda-leg left" d="M62 224c-14 4-25 14-22 25 4 13 27 12 41 0 8-7 3-29-19-25z" />
              <path className="panda-leg right" d="M158 224c14 4 25 14 22 25-4 13-27 12-41 0-8-7-3-29 19-25z" />
              <path className="panda-torso" d="M56 139c-18 25-20 74 5 96 23 21 74 21 98 0 25-22 23-71 5-96-19-26-89-26-108 0z" fill={`url(#pandaBody-${side})`} />
              <path className="panda-belly" d="M82 170c-15 16-14 48 0 61 14 13 42 13 56 0 14-13 15-45 0-61-14-15-42-15-56 0z" />
            </g>
            <path className="mascot-neck-svg panda-neck" d="M94 125c2 18 30 18 32 0v32c-4 13-28 13-32 0z" />
            <g className="mascot-head-svg">
              <circle className="panda-ear left" cx="67" cy="74" r="25" />
              <circle className="panda-ear right" cx="153" cy="74" r="25" />
              <path className="panda-head" d="M52 89c0-39 29-66 58-66s58 27 58 66c0 42-27 67-58 67s-58-25-58-67z" />
              <path className="panda-mask left" d="M65 88c-6-18 10-35 27-27 13 6 17 25 6 37-10 12-28 7-33-10z" />
              <path className="panda-mask right" d="M155 88c6-18-10-35-27-27-13 6-17 25-6 37 10 12 28 7 33-10z" />
              <g className="mascot-eye-svg left">
                <circle cx="84" cy="86" r="10" />
                <circle className="mascot-pupil" cx="84" cy="86" r="4" />
                <circle className="eye-shine" cx="81" cy="82" r="2" />
              </g>
              <g className="mascot-eye-svg right">
                <circle cx="136" cy="86" r="10" />
                <circle className="mascot-pupil" cx="136" cy="86" r="4" />
                <circle className="eye-shine" cx="133" cy="82" r="2" />
              </g>
              <path className="panda-nose" d="M103 111c4-5 10-5 14 0-1 7-13 7-14 0z" />
              <path className="panda-mouth" d="M110 119c-2 8-11 10-17 3M110 119c2 8 11 10 17 3" />
              <path className="face-blush left" d="M66 111c10-6 20-4 26 4" />
              <path className="face-blush right" d="M154 111c-10-6-20-4-26 4" />
            </g>
            <g className="mascot-eye-cover panda-cover-paws">
              <path className="cover-arm left" d="M61 158C49 133 58 106 78 91" />
              <path className="cover-arm right" d="M159 158C171 133 162 106 142 91" />
              <path className="cover-palm left" d="M64 91c0-14 10-24 25-25 15-1 27 8 29 21 2 14-9 23-26 25-17 1-28-7-28-21z" />
              <path className="cover-palm right" d="M156 91c0-14-10-24-25-25-15-1-27 8-29 21-2 14 9 23 26 25 17 1 28-7 28-21z" />
              <path className="cover-finger left" d="M75 78c7 4 17 4 25 0M72 89c9 5 24 5 35 0M77 101c8 3 17 3 25-1" />
              <path className="cover-finger right" d="M145 78c-7 4-17 4-25 0M148 89c-9 5-24 5-35 0M143 101c-8 3-17 3-25-1" />
              <path className="palm-highlight left" d="M80 71c8-3 19-1 26 7" />
              <path className="palm-highlight right" d="M140 71c-8-3-19-1-26 7" />
            </g>
            <g className="mascot-arm-svg left">
              <path className="panda-arm" d="M62 153c-26 15-38 39-29 51 10 13 34-2 46-27" />
              <circle className="panda-hand" cx="32" cy="204" r="18" />
            </g>
            <g className="mascot-arm-svg right">
              <path className="panda-arm" d="M158 153c26 15 38 39 29 51-10 13-34-2-46-27" />
              <circle className="panda-hand" cx="188" cy="204" r="18" />
            </g>
          </g>
        ) : (
          <g className="mascot-character buddy-character" filter={`url(#mascotShadow-${side})`}>
            <g className="mascot-body-svg">
              <path className="buddy-leg left" d="M67 222c-15 5-24 14-21 25 4 12 26 11 39 0 8-7 3-30-18-25z" />
              <path className="buddy-leg right" d="M153 222c15 5 24 14 21 25-4 12-26 11-39 0-8-7-3-30 18-25z" />
              <path className="buddy-torso" d="M57 139c-18 24-20 73 3 96 23 22 77 22 100 0 23-23 21-72 3-96-21-27-85-27-106 0z" fill={`url(#buddyBody-${side})`} />
              <path className="buddy-panel" d="M77 170c9-12 57-12 66 0 8 12 8 38 0 50-9 12-57 12-66 0-8-12-8-38 0-50z" />
              <path className="buddy-badge" d="M110 178l12 7v14c0 12-7 19-12 22-5-3-12-10-12-22v-14z" />
            </g>
            <path className="mascot-neck-svg buddy-neck" d="M93 125c1 19 33 19 34 0v34c-5 13-29 13-34 0z" />
            <g className="mascot-head-svg">
              <path className="buddy-antenna" d="M110 26v-18M96 12h28" />
              <circle className="buddy-antenna-dot" cx="110" cy="6" r="6" />
              <path className="buddy-head" d="M47 88c0-37 29-65 63-65s63 28 63 65c0 40-28 67-63 67s-63-27-63-67z" />
              <path className="buddy-goggle" d="M67 82c0-18 12-31 28-31h30c16 0 28 13 28 31s-12 31-28 31H95c-16 0-28-13-28-31z" />
              <g className="mascot-eye-svg left">
                <circle cx="92" cy="83" r="13" />
                <circle className="mascot-pupil" cx="92" cy="83" r="5" />
                <circle className="eye-shine" cx="88" cy="78" r="2.4" />
              </g>
              <g className="mascot-eye-svg right">
                <circle cx="128" cy="83" r="13" />
                <circle className="mascot-pupil" cx="128" cy="83" r="5" />
                <circle className="eye-shine" cx="124" cy="78" r="2.4" />
              </g>
              <path className="buddy-mouth" d="M91 120c10 12 28 12 38 0" />
              <path className="face-blush left" d="M65 111c10-5 19-3 25 3" />
              <path className="face-blush right" d="M155 111c-10-5-19-3-25 3" />
            </g>
            <g className="mascot-eye-cover buddy-cover-paws">
              <path className="cover-arm left" d="M61 158C50 131 65 101 86 86" />
              <path className="cover-arm right" d="M159 158C170 131 155 101 134 86" />
              <path className="cover-palm left" d="M68 83c0-15 12-26 28-26 17 0 29 10 29 24 0 14-12 23-30 24-17 1-27-8-27-22z" />
              <path className="cover-palm right" d="M152 83c0-15-12-26-28-26-17 0-29 10-29 24 0 14 12 23 30 24 17 1 27-8 27-22z" />
              <path className="cover-finger left" d="M80 70c8 4 19 4 29 0M76 82c11 5 26 5 38 0M81 95c9 3 19 3 28-1" />
              <path className="cover-finger right" d="M140 70c-8 4-19 4-29 0M144 82c-11 5-26 5-38 0M139 95c-9 3-19 3-28-1" />
              <path className="palm-highlight left" d="M84 62c10-3 22 0 29 9" />
              <path className="palm-highlight right" d="M136 62c-10-3-22 0-29 9" />
            </g>
            <g className="mascot-arm-svg left">
              <path className="buddy-arm" d="M62 153c-27 15-38 39-29 51 10 13 34-2 46-27" />
              <circle className="buddy-hand" cx="32" cy="204" r="18" />
            </g>
            <g className="mascot-arm-svg right">
              <path className="buddy-arm" d="M158 153c27 15 38 39 29 51-10 13-34-2-46-27" />
              <circle className="buddy-hand" cx="188" cy="204" r="18" />
            </g>
          </g>
        )}
      </svg>
    </aside>
  );
}

export function AuthPanel({ onAuthed }: Props) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [account, setAccount] = useState("");
  const [password, setPassword] = useState("");
  const [registerCode, setRegisterCode] = useState("");
  const [loginCaptchaCode, setLoginCaptchaCode] = useState("");
  const [loginCaptcha, setLoginCaptcha] = useState<LoginCaptchaResponse | null>(null);
  const [captchaRefreshing, setCaptchaRefreshing] = useState(false);
  const [message, setMessage] = useState("");
  const [messageType, setMessageType] = useState<FeedbackType>("info");
  const [busy, setBusy] = useState(false);
  const [countdown, setCountdown] = useState(0);
  const [showPassword, setShowPassword] = useState(false);
  const [focusField, setFocusField] = useState<FocusField>(null);
  const [shakeError, setShakeError] = useState(false);
  const [pointerLook, setPointerLook] = useState({ x: 0, y: 0 });

  const isLogin = mode === "login";
  const mascotMode: MascotMode = useMemo(() => {
    if (shakeError) return "error";
    if (focusField === "password") return "cover";
    if (focusField) return "peek";
    return "idle";
  }, [focusField, shakeError]);
  const mascotTarget: MascotTarget = useMemo(() => {
    if (shakeError) return "error";
    if (focusField === "account") return "account";
    if (focusField === "password") return "password";
    if (focusField === "login-captcha" || focusField === "register-code") return "captcha";
    return "idle";
  }, [focusField, shakeError]);
  const captchaChars = useMemo(() => (loginCaptcha?.captcha_text ?? "-----").split(""), [loginCaptcha]);
  const heading = isLogin ? "登录分析平台" : "注册分析账号";
  const passwordLabel = isLogin ? "登录密码" : "设置密码";
  const submitLabel = busy ? "正在验证" : isLogin ? "进入工作台" : "创建账号并进入";

  function getMascotStyle(side: "left" | "right"): CSSProperties {
    const targetLook: Record<MascotTarget, { x: number; y: number }> = {
      idle: pointerLook,
      account: { x: side === "left" ? 0.9 : -0.9, y: -0.58 },
      password: { x: side === "left" ? 0.35 : -0.35, y: -0.12 },
      captcha: { x: side === "left" ? 1 : -1, y: 0.55 },
      error: { x: 0, y: 0.25 }
    };
    const look = targetLook[mascotTarget];
    return {
      "--look-x": `${look.x * 8}px`,
      "--look-y": `${look.y * 5}px`,
      "--look-rot": `${look.x * 6}deg`,
      "--char-rot": `${look.x * -2.3}deg`,
      "--body-rot": `${look.x * -1.8}deg`,
      "--pupil-x": `${look.x * 6}px`,
      "--pupil-y": `${look.y * 4}px`,
      "--lean-x": `${look.x * 10}px`,
      "--lean-y": `${look.y * 5}px`,
      "--neck-x": `${look.x * 3}px`,
      "--neck-y": `${look.y * 2}px`,
      "--body-x": `${look.x * -1.2}px`,
      "--body-y": `${look.y * 0.8}px`
    } as CSSProperties;
  }

  function handleStagePointerMove(event: PointerEvent<HTMLElement>) {
    if (focusField) return;
    const width = Math.max(window.innerWidth, 1);
    const height = Math.max(window.innerHeight, 1);
    const x = Math.max(-1, Math.min(1, (event.clientX / width - 0.5) * 2));
    const y = Math.max(-1, Math.min(1, (event.clientY / height - 0.45) * 2));
    setPointerLook({ x, y });
  }

  function showMessage(text: string, type: FeedbackType = "info") {
    setMessage(text);
    setMessageType(type);
  }

  useEffect(() => {
    if (!isLogin) return;
    refreshLoginCaptcha().catch(() => {
      showMessage("登录验证码加载失败，请稍后重试。", "error");
    });
  }, [isLogin]);

  useEffect(() => {
    if (countdown <= 0) return;
    const timer = window.setInterval(() => {
      setCountdown((value) => {
        if (value <= 1) {
          window.clearInterval(timer);
          return 0;
        }
        return value - 1;
      });
    }, 1000);
    return () => window.clearInterval(timer);
  }, [countdown]);

  useEffect(() => {
    if (!shakeError) return;
    const timer = window.setTimeout(() => setShakeError(false), 1400);
    return () => window.clearTimeout(timer);
  }, [shakeError]);

  async function refreshLoginCaptcha() {
    setCaptchaRefreshing(true);
    try {
      const result = await fetchLoginCaptcha();
      setLoginCaptcha(result);
      setLoginCaptchaCode("");
    } catch (error) {
      showMessage(error instanceof Error ? error.message : "登录验证码加载失败，请稍后重试。", "error");
      triggerMascotError();
      throw error;
    } finally {
      setCaptchaRefreshing(false);
    }
  }

  async function handleRegisterCode() {
    if (!account.trim()) {
      showMessage("请先输入邮箱账号。", "error");
      triggerMascotError();
      return;
    }
    setBusy(true);
    try {
      const result = await requestCode(account);
      setRegisterCode("");
      showMessage(result.code ? "邮箱验证码已发送，请查收邮件。" : result.message, "success");
      setCountdown(60);
    } catch (error) {
      showMessage(error instanceof Error ? error.message : "邮箱验证码发送失败，请稍后重试。", "error");
      triggerMascotError();
    } finally {
      setBusy(false);
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    try {
      const result = isLogin
        ? await login(account, password, loginCaptcha?.captcha_id ?? "", loginCaptchaCode)
        : await register(account, password, registerCode);
      showMessage(result.message, "success");
      onAuthed(result);
    } catch (error) {
      showMessage(error instanceof Error ? error.message : "认证失败，请检查输入后重试。", "error");
      triggerMascotError();
      if (isLogin) {
        refreshLoginCaptcha().catch(() => undefined);
      }
    } finally {
      setBusy(false);
    }
  }

  function triggerMascotError() {
    setShakeError(true);
  }

  function switchMode(nextMode: "login" | "register") {
    setMode(nextMode);
    setRegisterCode("");
    setLoginCaptchaCode("");
    setCountdown(0);
    setFocusField(null);
    setShowPassword(false);
    showMessage("", "info");
  }

  return (
    <main className="auth-stage" onPointerMove={handleStagePointerMove} onPointerLeave={() => setPointerLook({ x: 0, y: 0 })}>
      <div className="auth-background-grid" />
      <div className="auth-noise" />
      <div className="auth-constellation auth-constellation-one" />
      <div className="auth-constellation auth-constellation-two" />
      <div className="auth-glow auth-glow-one" />
      <div className="auth-glow auth-glow-two" />
      <div className="auth-glow auth-glow-three" />

      <section className="auth-shell">
        <header className="auth-topbar">
          <div className="brand-mark auth-brand">星</div>
          <p className="eyebrow">异常交易识别与关系图谱分析</p>
          <h1>星枢反欺诈分析平台</h1>
        </header>

        <section className="auth-main-layout">
          <AuthMascot side="left" mode={mascotMode} target={mascotTarget} style={getMascotStyle("left")} />

          <section className={`auth-card auth-form-card glass-panel ${shakeError ? "card-shake" : ""}`}>
            <div className="auth-card-header compact-header">
              <div>
                <p className="auth-card-kicker">欢迎使用</p>
                <h2>{heading}</h2>
              </div>
              <div className="segmented auth-mode-switch" aria-label="认证模式切换">
                <button type="button" className={isLogin ? "active" : ""} onClick={() => switchMode("login")}>
                  登录
                </button>
                <button type="button" className={!isLogin ? "active" : ""} onClick={() => switchMode("register")}>
                  注册
                </button>
              </div>
            </div>

            <form className="auth-form minimal-form" onSubmit={handleSubmit}>
              <label htmlFor="auth-account">
                邮箱账号
                <div className="input-shell">
                  <input
                    id="auth-account"
                    autoComplete="email"
                    value={account}
                    onChange={(event) => setAccount(event.target.value)}
                    onFocus={() => setFocusField("account")}
                    onBlur={() => setFocusField(null)}
                    placeholder="name@example.com"
                  />
                </div>
              </label>

              <label htmlFor="auth-password">
                {passwordLabel}
                <div className="input-shell with-action">
                  <input
                    id="auth-password"
                    autoComplete={isLogin ? "current-password" : "new-password"}
                    type={showPassword ? "text" : "password"}
                    value={password}
                    onChange={(event) => setPassword(event.target.value)}
                    onFocus={() => setFocusField("password")}
                    onBlur={() => setFocusField(null)}
                    placeholder={isLogin ? "请输入密码" : "请设置密码"}
                  />
                  <button
                    type="button"
                    className="input-action eye-action"
                    onClick={() => setShowPassword((value) => !value)}
                    aria-label={showPassword ? "隐藏密码" : "显示密码"}
                    title={showPassword ? "隐藏密码" : "显示密码"}
                  >
                    <EyeIcon open={showPassword} />
                  </button>
                </div>
              </label>

              {isLogin ? (
                <div className="login-captcha-grid">
                  <label htmlFor="login-captcha-code">
                    验证码
                    <div className="input-shell">
                      <input
                        id="login-captcha-code"
                        inputMode="text"
                        value={loginCaptchaCode}
                        onChange={(event) => setLoginCaptchaCode(event.target.value.toUpperCase())}
                        onFocus={() => setFocusField("login-captcha")}
                        onBlur={() => setFocusField(null)}
                        placeholder="输入字符"
                      />
                    </div>
                  </label>

                  <button type="button" className="captcha-card captcha-code" title="点击刷新验证码" onClick={() => refreshLoginCaptcha()} disabled={captchaRefreshing || busy} aria-label="登录验证码">
                    {captchaChars.map((char, index) => (
                      <span key={`${char}-${index}`}>{char}</span>
                    ))}
                  </button>
                  <button
                    type="button"
                    className={`captcha-refresh icon-refresh ${captchaRefreshing ? "spinning" : ""}`}
                    onClick={() => refreshLoginCaptcha()}
                    disabled={captchaRefreshing || busy}
                    aria-label="刷新验证码"
                    title="刷新验证码"
                  >
                    <RefreshIcon />
                  </button>
                </div>
              ) : (
                <div className="captcha-row register-row">
                  <label htmlFor="register-code">
                    邮箱验证码
                    <div className="input-shell">
                      <input
                        id="register-code"
                        inputMode="numeric"
                        value={registerCode}
                        onChange={(event) => setRegisterCode(event.target.value)}
                        onFocus={() => setFocusField("register-code")}
                        onBlur={() => setFocusField(null)}
                        placeholder="请输入邮箱验证码"
                      />
                    </div>
                  </label>
                  <button type="button" className="secondary register-code-button" onClick={handleRegisterCode} disabled={busy || countdown > 0}>
                    {countdown > 0 ? `${countdown}s 后重试` : "发送邮箱验证码"}
                  </button>
                </div>
              )}

              <button className="primary auth-submit" type="submit" disabled={busy}>
                {submitLabel}
              </button>
            </form>

            {message ? <p className={`hint auth-feedback minimal-feedback ${messageType}`}>{message}</p> : null}
          </section>

          <AuthMascot side="right" mode={mascotMode} target={mascotTarget} style={getMascotStyle("right")} />
        </section>

        <footer className="auth-footer-facts">
          {footerFacts.map((item) => (
            <span key={item}>{item}</span>
          ))}
        </footer>
      </section>
    </main>
  );
}
