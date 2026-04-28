import { FormEvent, useEffect, useMemo, useState } from "react";
import { AuthResponse, fetchLoginCaptcha, LoginCaptchaResponse, login, register, requestCode } from "../services/api";

type Props = {
  onAuthed: (session: AuthResponse) => void;
};

type FocusField = "account" | "password" | "register-code" | "login-captcha" | null;
type MascotMode = "idle" | "peek" | "cover" | "error";

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

function AuthMascot({ side, mode }: { side: "left" | "right"; mode: MascotMode }) {
  const label = side === "left" ? "账号观察员" : "风险守卫员";

  return (
    <aside className={`mascot-stage mascot-${side} mascot-${mode}`} aria-hidden="true">
      <div className="mascot-orbit">
        <span />
        <span />
        <span />
      </div>
      <div className="mascot-figure">
        <div className="mascot-neck" />
        <div className="mascot-head">
          <div className="mascot-ear left" />
          <div className="mascot-ear right" />
          <div className="mascot-face-shine" />
          <div className="mascot-eye left">
            <span />
          </div>
          <div className="mascot-eye right">
            <span />
          </div>
          <div className="mascot-nose" />
          <div className="mascot-mouth" />
        </div>
        <div className="mascot-body">
          <div className="mascot-collar" />
          <div className="mascot-core" />
          <div className="mascot-arm left">
            <span className="mascot-hand" />
          </div>
          <div className="mascot-arm right">
            <span className="mascot-hand" />
          </div>
        </div>
      </div>
      <div className="mascot-shadow" />
      <div className="mascot-label">{label}</div>
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
  const [busy, setBusy] = useState(false);
  const [countdown, setCountdown] = useState(0);
  const [showPassword, setShowPassword] = useState(false);
  const [focusField, setFocusField] = useState<FocusField>(null);
  const [shakeError, setShakeError] = useState(false);

  const isLogin = mode === "login";
  const mascotMode: MascotMode = useMemo(() => {
    if (shakeError) return "error";
    if (focusField === "password") return "cover";
    if (focusField) return "peek";
    return "idle";
  }, [focusField, shakeError]);
  const captchaChars = useMemo(() => (loginCaptcha?.captcha_text ?? "-----").split(""), [loginCaptcha]);
  const heading = isLogin ? "登录分析平台" : "注册分析账号";
  const passwordLabel = isLogin ? "登录密码" : "设置密码";
  const submitLabel = busy ? "正在验证" : isLogin ? "进入工作台" : "创建账号并进入";

  useEffect(() => {
    if (!isLogin) return;
    refreshLoginCaptcha().catch(() => {
      setMessage("验证码加载失败，请稍后重试。");
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
      setMessage(error instanceof Error ? error.message : "验证码加载失败，请稍后重试。");
      triggerMascotError();
      throw error;
    } finally {
      setCaptchaRefreshing(false);
    }
  }

  async function handleRegisterCode() {
    if (!account.trim()) {
      setMessage("请输入邮箱账号。");
      triggerMascotError();
      return;
    }
    setBusy(true);
    try {
      const result = await requestCode(account);
      setRegisterCode("");
      setMessage(result.code ? "验证码已发送，请查看邮箱。" : result.message);
      setCountdown(60);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "验证码发送失败，请稍后重试。");
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
      setMessage(result.message);
      onAuthed(result);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "认证失败。");
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
    setMessage("");
  }

  return (
    <main className="auth-stage">
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
          <AuthMascot side="left" mode={mascotMode} />

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

            {message ? <p className="hint auth-feedback minimal-feedback">{message}</p> : null}
          </section>

          <AuthMascot side="right" mode={mascotMode} />
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
