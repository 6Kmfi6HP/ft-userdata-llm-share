# 配置文件说明

## 重要提示

在使用本项目前，您需要配置以下敏感信息。**请勿将包含真实密钥的配置文件提交到版本控制系统！**

## 配置步骤

### 1. LLM API 配置

在 `user_data/config.json` 的 `llm_config` 部分填写：

```json
"llm_config": {
    "api_base": "http://localhost:3120",
    "api_key": "your-api-key-here",
    "model": "qwen/qwen3-coder-30b",
    "embedding_model": "text-embedding-bge-m3"
}
```

### 2. 交易所 API 配置

本策略支持 **Binance** 和 **Hyperliquid** 交易所。

#### Binance (默认)

```json
"stake_currency": "USDT",
"exchange": {
    "name": "binance",
    "key": "your-binance-api-key",
    "secret": "your-binance-api-secret",
    "ccxt_config": {
        "enableRateLimit": true,
        "options": {
            "defaultType": "future"
        }
    },
    "ccxt_async_config": {
        "enableRateLimit": true,
        "rateLimit": 200,
        "timeout": 30000
    }
}
```

**Binance API 设置步骤：**

1. 访问 [Binance API Management](https://www.binance.com/en/my/settings/api-management)
2. 创建新的 API 密钥
3. 启用 **Futures** 权限
4. 限制 IP 地址访问（推荐）
5. 将 API key 和 secret 复制到配置文件

#### Hyperliquid

```json
"stake_currency": "USDC",
"exchange": {
    "name": "hyperliquid",
    "walletAddress": "0x<your-wallet-address-40-hex-chars>",
    "privateKey": "0x<your-api-wallet-private-key-64-hex-chars>",
    "ccxt_config": {
        "enableRateLimit": true
    },
    "ccxt_async_config": {
        "enableRateLimit": true
    }
}
```

**Hyperliquid API 设置步骤：**

1. 访问 [Hyperliquid App](https://app.hyperliquid.xyz/)
2. 连接你的钱包
3. 进入 **Settings** > **API Wallet**
4. 创建新的 API 钱包（系统会生成一个独立的交易钱包）
5. 将 **钱包地址** 和 **私钥** 复制到配置文件

**重要提示：**

- 使用 **API 钱包** 凭证，而非主钱包
- API 钱包权限受限，更安全
- 需要向 API 钱包充值 USDC 才能交易

#### 交易所对比

| 特性 | Binance | Hyperliquid |
| ---- | ------- | ----------- |
| 结算货币 | USDT | USDC |
| 认证方式 | API key + secret | 钱包地址 + 私钥 |
| 交易对格式 | `BTC/USDT:USDT` | `BTC/USDC:USDC` |
| 历史K线 | 无限制 | 最多5000根 |
| 市价单 | 原生支持 | 限价单模拟（5%滑点） |
| 配置字段 | `key`, `secret` | `walletAddress`, `privateKey` |

**安全建议：**

- 使用子账户进行测试
- 限制 API 权限（仅开启交易权限，不开启提现权限）
- 建议先使用 `"dry_run": true` 模式进行模拟交易

### 3. 通知配置（可选）

#### Telegram 通知

```json
"telegram": {
    "enabled": true,
    "token": "your-telegram-bot-token",
    "chat_id": "your-telegram-chat-id"
}
```

#### Discord 通知

```json
"discord": {
    "enabled": true,
    "webhook_url": "your-discord-webhook-url",
    ...
}
```

### 4. API Server 配置

```json
"api_server": {
    "enabled": true,
    "jwt_secret_key": "your-jwt-secret-key",
    "username": "your-username",
    "password": "your-password"
}
```

生成安全的 JWT 密钥：

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## 环境变量方式（推荐）

您也可以使用环境变量来配置敏感信息：

1. 复制 `.env.example` 为 `.env`
2. 在 `.env` 中填写实际值
3. 修改代码以支持从环境变量读取

## 安全检查清单

在分享代码或提交到 GitHub 前，请确保：

- [ ] 所有 API 密钥已移除或替换为占位符
- [ ] 交易所密钥已清空
- [ ] JWT 密钥已清空
- [ ] Telegram/Discord 令牌已清空
- [ ] 日志文件不包含敏感信息
- [ ] 数据库文件未包含在版本控制中
- [ ] `.gitignore` 已正确配置
- [ ] `.env` 文件未被提交

## 测试配置

配置完成后，建议先使用模拟模式测试：

```bash
# 确保 config.json 中设置了：
"dry_run": true,
"dry_run_wallet": 1340,
```

这样可以在不使用真实资金的情况下测试策略。

## 获取帮助

如果您在配置过程中遇到问题，请查看：

- Freqtrade 官方文档：<https://www.freqtrade.io/>
- 本项目的 README.md 文件
