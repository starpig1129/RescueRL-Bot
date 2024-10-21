using UnityEngine;
using System;
using System.Net.Sockets;
using System.IO;

public class CameraCapture : MonoBehaviour
{
    public Camera captureCamera;
    private TcpClient client;
    private NetworkStream stream;
    private Texture2D texture;
    private byte[] imageBytes;

    void Start()
    {
        if (captureCamera == null)
        {
            Debug.LogError("Capture Camera is not assigned!");
            this.enabled = false;
            return;
        }

        try
        {
            client = new TcpClient("localhost", 6000);
            stream = client.GetStream();
        }
        catch (SocketException e)
        {
            Debug.LogError("Socket exception: " + e.ToString());
            this.enabled = false;
            return;
        }

        RenderTexture renderTexture = captureCamera.targetTexture;
        if (renderTexture != null)
        {
            texture = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false);
        }

        InvokeRepeating("SendImage", 0.03f, 0.03f); // Send image every 0.1 second
    }

    void SendImage()
    {
        if (texture != null && stream != null)
        {
            RenderTexture renderTexture = captureCamera.targetTexture;
            RenderTexture.active = renderTexture;
            texture.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
            texture.Apply();

            imageBytes = texture.EncodeToJPG();

            try
            {
                stream.Write(BitConverter.GetBytes(imageBytes.Length), 0, 4);
                stream.Write(imageBytes, 0, imageBytes.Length);
            }
            catch (IOException e)
            {
                Debug.LogError("Failed to send image: " + e.Message);
                // Optionally implement reconnection logic here
            }
        }
    }

    void OnDestroy()
    {
        if (stream != null)
        {
            stream.Close();
        }
        if (client != null)
        {
            client.Close();
        }
    }
}
