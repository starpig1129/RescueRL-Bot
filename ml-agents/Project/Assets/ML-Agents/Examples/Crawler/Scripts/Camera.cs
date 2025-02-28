using UnityEngine;
using System;
using System.Threading.Tasks;

public class CameraCapture : MonoBehaviour
{
    public Camera captureCamera;
    private Texture2D texture;
    private float captureInterval = 0.03f;
    private bool isCapturing = true;

    void Start()
    {
        if (captureCamera == null)
        {
            Debug.LogError("Capture Camera is not assigned!");
            this.enabled = false;
            return;
        }

        RenderTexture renderTexture = captureCamera.targetTexture;
        if (renderTexture != null)
        {
            texture = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false);
        }

        // 確保CommunicationManager實例存在
        var manager = CommunicationManager.Instance;

        // 使用協程來定期捕獲和發送圖像
        StartCoroutine(CaptureAndSendRoutine());
    }

    System.Collections.IEnumerator CaptureAndSendRoutine()
    {
        while (isCapturing)
        {
            yield return new WaitForSeconds(captureInterval);

            if (texture == null) continue;

            Debug.Log($"開始obs觀測");

            // 捕獲圖像
            RenderTexture renderTexture = captureCamera.targetTexture;
            RenderTexture.active = renderTexture;
            texture.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
            texture.Apply();

            // 使用通信管理器發送圖像
            SendImageAsync(texture).ContinueWith(task => {
                if (task.Exception != null)
                {
                    Debug.LogError($"發送相機圖像時發生錯誤: {task.Exception.Message}");
                }
                else
                {
                    Debug.Log($"obs傳送成功");
                }
            });
        }
    }

    private async Task<bool> SendImageAsync(Texture2D texture)
    {
        try
        {
            // 使用通信管理器發送圖像
            return await CommunicationManager.Instance.SendImageAsync(
                texture,
                CommunicationManager.Instance.ObsStream,
                CommunicationManager.Instance.ObsLock,
                CommunicationManager.Instance.IsObsConnected
            );
        }
        catch (Exception e)
        {
            Debug.LogError($"發送相機圖像時發生錯誤: {e.Message}");
            return false;
        }
    }

    void OnDestroy()
    {
        isCapturing = false;
    }
}
