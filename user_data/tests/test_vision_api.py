#!/usr/bin/env python3
"""
测试视觉分析 API 调用

用于诊断为什么视觉分析返回空响应
"""
import base64
import json
import os
import sys
import requests
from io import BytesIO

# 添加项目路径
sys.path.insert(0, '/freqtrade/user_data/strategies')


def create_test_image():
    """创建一个简单的测试图片 (红色方块)"""
    try:
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='red')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    except ImportError:
        # 如果没有 PIL，使用一个最小的有效PNG
        # 这是一个1x1像素的红色PNG
        minimal_png = (
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
            b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00'
            b'\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03'
            b'\x00\x01\x00\x05\xfe\xd4\x00\x00\x00\x00IEND\xaeB`\x82'
        )
        return base64.b64encode(minimal_png).decode('utf-8')


def test_vision_api_v1(api_base: str, api_key: str, model: str):
    """测试方法1: 标准 OpenAI 格式"""
    print("\n" + "=" * 60)
    print("测试方法1: 标准 OpenAI vision 格式")
    print("=" * 60)

    image_base64 = create_test_image()

    url = f"{api_base}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "这张图片是什么颜色？请用一个词回答。"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 100,
        "temperature": 0.1
    }

    print(f"API URL: {url}")
    print(f"Model: {model}")
    print(f"Image size: {len(image_base64)} bytes (base64)")
    print(f"Payload (truncated): {json.dumps(payload, ensure_ascii=False)[:500]}...")

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")

        if response.status_code == 200:
            data = response.json()
            print(f"\nFull Response JSON:\n{json.dumps(data, indent=2, ensure_ascii=False)}")

            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"\n✅ Content: '{content}'")
            print(f"Content length: {len(content)} chars")
        else:
            print(f"\n❌ Error Response:\n{response.text}")

    except Exception as e:
        print(f"\n❌ Exception: {e}")


def test_vision_api_v2_streaming(api_base: str, api_key: str, model: str):
    """测试方法2: 用户提供的流式格式 (Google specific)"""
    print("\n" + "=" * 60)
    print("测试方法2: 流式格式 (stream=true)")
    print("=" * 60)

    image_base64 = create_test_image()

    url = f"{api_base}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "这张图片是什么颜色？请用一个词回答。"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "stream": True,
        "stream_options": {
            "include_usage": True
        }
    }

    print(f"API URL: {url}")
    print(f"Model: {model}")
    print(f"Stream: True")

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30, stream=True)
        print(f"\nStatus Code: {response.status_code}")

        if response.status_code == 200:
            print("\n--- Streaming Response ---")
            full_content = ""
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    print(f"Line: {line_str[:200]}")
                    if line_str.startswith("data: ") and line_str != "data: [DONE]":
                        try:
                            chunk = json.loads(line_str[6:])
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_content += content
                        except json.JSONDecodeError:
                            pass
            print(f"\n✅ Full Content: '{full_content}'")
        else:
            print(f"\n❌ Error Response:\n{response.text}")

    except Exception as e:
        print(f"\n❌ Exception: {e}")


def test_vision_api_v3_no_stream(api_base: str, api_key: str, model: str):
    """测试方法3: 非流式但带 Google extra_body"""
    print("\n" + "=" * 60)
    print("测试方法3: 非流式 + extra_body (如果API支持)")
    print("=" * 60)

    image_base64 = create_test_image()

    url = f"{api_base}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this image in one word."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
    }

    print(f"Using English prompt to test language support")

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"\nStatus Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"\nFull Response:\n{json.dumps(data, indent=2, ensure_ascii=False)}")

            # 检查不同的响应格式
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                # 有些模型可能用不同的字段
                if not content:
                    content = message.get("text", "")
                if not content and isinstance(message.get("content"), list):
                    # content 可能是数组
                    for item in message.get("content", []):
                        if isinstance(item, dict) and item.get("type") == "text":
                            content = item.get("text", "")
                            break
                        elif isinstance(item, str):
                            content = item
                            break

                print(f"\n✅ Extracted Content: '{content}'")
            else:
                print("\n❌ No choices in response")
        else:
            print(f"\n❌ Error Response:\n{response.text}")

    except Exception as e:
        print(f"\n❌ Exception: {e}")


def test_text_only(api_base: str, api_key: str, model: str):
    """测试纯文本调用 (确认API基本工作)"""
    print("\n" + "=" * 60)
    print("测试方法0: 纯文本调用 (baseline)")
    print("=" * 60)

    url = f"{api_base}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Say 'Hello' in Chinese. One word only."}
        ],
        "max_tokens": 50,
        "temperature": 0.1
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"✅ Text Response: '{content}'")
        else:
            print(f"❌ Error: {response.text}")

    except Exception as e:
        print(f"❌ Exception: {e}")


def test_with_real_chart():
    """测试使用真实的K线图"""
    print("\n" + "=" * 60)
    print("测试方法4: 使用真实 K线图生成")
    print("=" * 60)

    try:
        # 尝试导入 chart_generator
        from llm_modules.utils.chart_generator import ChartGenerator
        import pandas as pd
        import numpy as np

        # 创建模拟的 OHLCV 数据
        np.random.seed(42)
        n = 50
        dates = pd.date_range(start='2024-01-01', periods=n, freq='30min')
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        open_price = close + np.random.randn(n) * 0.2
        volume = np.random.randint(1000, 10000, n)

        df = pd.DataFrame({
            'date': dates,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        df.set_index('date', inplace=True)

        # 生成图表 - 使用正确的方法名
        generator = ChartGenerator()

        # 测试 K线图生成
        print("\n--- 测试 generate_kline_image ---")
        kline_result = generator.generate_kline_image(df, pair="TEST/USDT", timeframe="30m")
        if kline_result.get("success"):
            kline_image = kline_result["image_base64"]
            print(f"✅ K线图生成成功: {len(kline_image)} bytes (base64)")
            # 保存图片
            import base64
            image_bytes = base64.b64decode(kline_image)
            with open('/tmp/test_kline.png', 'wb') as f:
                f.write(image_bytes)
            print("   图片已保存到 /tmp/test_kline.png")
        else:
            print(f"❌ K线图生成失败: {kline_result.get('error')}")
            kline_image = None

        # 测试趋势线图生成
        print("\n--- 测试 generate_trend_image ---")
        trend_result = generator.generate_trend_image(df, pair="TEST/USDT", timeframe="30m")
        if trend_result.get("success"):
            trend_image = trend_result["image_base64"]
            print(f"✅ 趋势线图生成成功: {len(trend_image)} bytes (base64)")
            # 保存图片
            image_bytes = base64.b64decode(trend_image)
            with open('/tmp/test_trend.png', 'wb') as f:
                f.write(image_bytes)
            print("   图片已保存到 /tmp/test_trend.png")
        else:
            print(f"❌ 趋势线图生成失败: {trend_result.get('error')}")
            trend_image = None

        return kline_image or trend_image

    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_parallel_vision_calls(api_base: str, api_key: str, model: str):
    """测试并行 vision 调用（模拟 PatternAgent 和 TrendAgent 同时调用）"""
    print("\n" + "=" * 60)
    print("测试方法6: 并行 Vision API 调用")
    print("=" * 60)

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    image_base64 = create_test_image()

    def make_vision_call(call_id: str):
        url = f"{api_base}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Call {call_id}: 这张图片是什么颜色？"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                }
            ],
            "max_tokens": 50,
            "temperature": 0.1
        }

        start = time.time()
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            elapsed = time.time() - start
            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return call_id, True, content, elapsed
            else:
                return call_id, False, response.text, elapsed
        except Exception as e:
            return call_id, False, str(e), time.time() - start

    print("发起两个并行 vision 调用...")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(make_vision_call, "PatternAgent"),
            executor.submit(make_vision_call, "TrendAgent")
        ]

        for future in as_completed(futures):
            call_id, success, result, elapsed = future.result()
            if success:
                print(f"✅ {call_id}: '{result}' ({elapsed:.2f}s)")
            else:
                print(f"❌ {call_id}: {result} ({elapsed:.2f}s)")


def main():
    """主测试函数"""
    print("=" * 60)
    print("视觉 API 诊断测试")
    print("=" * 60)

    # 从环境变量或配置文件加载配置
    try:
        import json
        config_path = '/freqtrade/user_data/config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)

        llm_config = config.get('llm_config', {})
        api_base = llm_config.get('api_base', 'http://192.168.8.225:3899')
        api_key = llm_config.get('api_key', 'sk-xxx')
        model = llm_config.get('model', 'gemini-flash-lite-latest')

        print(f"API Base: {api_base}")
        print(f"Model: {model}")
        print(f"API Key: {api_key[:10]}..." if api_key else "No API Key")

    except Exception as e:
        print(f"加载配置失败: {e}")
        # 使用默认值
        api_base = os.environ.get('LLM_API_BASE', 'http://192.168.8.225:3899')
        api_key = os.environ.get('LLM_API_KEY', 'sk-xxx')
        model = os.environ.get('LLM_MODEL', 'gemini-flash-lite-latest')

    # 运行测试
    test_text_only(api_base, api_key, model)
    test_vision_api_v1(api_base, api_key, model)
    test_vision_api_v3_no_stream(api_base, api_key, model)

    # 测试并行调用（关键！模拟 PatternAgent + TrendAgent 同时调用）
    test_parallel_vision_calls(api_base, api_key, model)

    # 测试真实图表
    real_chart = test_with_real_chart()
    if real_chart:
        print("\n" + "=" * 60)
        print("测试方法5: 使用真实图表调用 Vision API")
        print("=" * 60)

        url = f"{api_base}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "这是一张K线图。请简单描述你看到了什么。"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{real_chart}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 200,
            "temperature": 0.1
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            print(f"Status Code: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"\n✅ Vision API 响应: {content[:500]}...")
                print(f"响应长度: {len(content)} 字符")
            else:
                print(f"❌ Error: {response.text}")

        except Exception as e:
            print(f"❌ Exception: {e}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
