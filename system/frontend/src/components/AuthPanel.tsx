import { FormEvent, useState } from "react";
import { login, register, requestCode } from "../services/api";

type Props = {
  onAuthed: (account: string) => void;
};

export function AuthPanel({ onAuthed }: Props) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [account, setAccount] = useState("root");
  const [password, setPassword] = useState("root");
  const [code, setCode] = useState("");
  const [captcha, setCaptcha] = useState("点击刷新");
  const [message, setMessage] = useState("默认账号 root，密码 root。请先获取右侧验证码。");
  const [busy, setBusy] = useState(false);

  async function handleCode() {
    setBusy(true);
    try {
      const result = await requestCode(account, mode);
      setCaptcha(result.code ?? "已发送");
      setCode("");
      setMessage(result.code ? "请按右侧验证码输入。" : result.message);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "验证码生成失败。");
    } finally {
      setBusy(false);
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    try {
      const result = mode === "register" ? await register(account, password, code) : await login(account, password, code);
      setMessage(result.message);
      onAuthed(result.email);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "认证失败。");
    } finally {
      setBusy(false);
    }
  }

  function switchMode(nextMode: "login" | "register") {
    setMode(nextMode);
    setCode("");
    setCaptcha("点击刷新");
    setMessage(nextMode === "login" ? "输入账号并按右侧验证码登录。" : "填写账号、密码和右侧验证码创建账号。");
  }

  return (
    <main className="auth-page">
      <section className="auth-intro">
        <div className="brand-mark">D</div>
        <p className="eyebrow">DyRIFT-TGAT</p>
        <h1>动态图欺诈节点推理系统</h1>
        <p>
          面向动态图异常检测的落地演示系统。登录后可上传交易数据、查看图关系、
          启动特征处理，并使用 full 模型权重完成异常节点推理。
        </p>
        <div className="auth-stats">
          <span>Pure GNN</span>
          <span>AUC Tracking</span>
          <span>Graph Inference</span>
        </div>
      </section>

      <section className="auth-card">
        <div className="auth-card-header">
          <div>
            <p className="eyebrow">User Access</p>
            <h2>{mode === "login" ? "登录系统" : "注册账号"}</h2>
          </div>
          <div className="segmented">
            <button type="button" className={mode === "login" ? "active" : ""} onClick={() => switchMode("login")}>
              登录
            </button>
            <button type="button" className={mode === "register" ? "active" : ""} onClick={() => switchMode("register")}>
              注册
            </button>
          </div>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          <label>
            账号
            <input value={account} onChange={(event) => setAccount(event.target.value)} />
          </label>
          <label>
            密码
            <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} />
          </label>
          <div className="captcha-row">
            <label>
              验证码
              <input value={code} onChange={(event) => setCode(event.target.value)} />
            </label>
            <button type="button" className="captcha-card" onClick={handleCode} disabled={busy}>
              <small>验证码</small>
              <strong>{captcha}</strong>
            </button>
          </div>
          <button className="primary auth-submit" type="submit" disabled={busy}>
            {mode === "login" ? "进入系统" : "创建并进入"}
          </button>
        </form>

        <p className="hint">{message}</p>
      </section>
    </main>
  );
}
