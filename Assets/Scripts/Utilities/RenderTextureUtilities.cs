using UnityEngine;
using UnityEngine.Rendering;

namespace StableFluids
{
    /// <summary>
    /// A collection of custom utility methods and classes for working
    /// with render textures.
    /// </summary>
    public class RenderTextureUtilities
    {
        public class CPURenderTexture2D
        {
            private Vector2[] _rawTextureData = null;
            private int _width = 0;
            private int _height = 0;

            public Vector2[] RawTextureData => _rawTextureData;
            public int Width => _width;
            public int Height => _height;

            public CPURenderTexture2D(int width, int height)
            {
                _width = width;
                _height = height;
                _rawTextureData = new Vector2[_width * _height];
            }

            public void ProcessGPUReadbackRequest(AsyncGPUReadbackRequest readbackRequest)
            {
                readbackRequest.GetData<Vector2>().CopyTo(_rawTextureData);
            }

            public Vector2 GetPixel(int x, int y)
            {
                return _rawTextureData[x + y * _width];
            }

            public Vector2 GetPixelBilinear(float u, float v)
            {
                int x1 = Mathf.FloorToInt(u * (_width - 1));
                int x2 = Mathf.CeilToInt(u * (_width - 1));
                int y1 = Mathf.FloorToInt(v * (_height - 1));
                int y2 = Mathf.CeilToInt(v * (_height - 1));

                Vector2 Q11 = GetPixel(x1, y1);
                Vector2 Q12 = GetPixel(x1, y2);
                Vector2 Q21 = GetPixel(x2, y1);
                Vector2 Q22 = GetPixel(x2, y2);

                Vector2 Q1112 = Vector2.Lerp(Q11, Q12, u - x1);
                Vector2 Q2122 = Vector2.Lerp(Q21, Q22, u - x1);

                return Vector2.Lerp(Q1112, Q2122, v - y1);
            }
        }
        
        public class CPURenderTexture3D
        {
            private Vector4[][] _rawTextureData = null;
            private int _width = 0;
            private int _height = 0;
            private int _depth = 0;

            public Vector4[][] RawTextureData => _rawTextureData;
            public int Width => _width;
            public int Height => _height;
            public int Depth => _depth;

            public CPURenderTexture3D(int width, int height, int depth)
            {
                _width = width;
                _height = height;
                _depth = depth;
                _rawTextureData = new Vector4[depth][];
                for (int i = 0; i < _depth; i++)
                {
                    _rawTextureData[i] = new Vector4[_width * height];
                }
            }

            public void ProcessGPUReadbackRequest(AsyncGPUReadbackRequest readbackRequest)
            {
                for (int i = 0; i < _depth; i++)
                {
                    readbackRequest.GetData<Vector4>(i).CopyTo(_rawTextureData[i]);
                }
            }

            public Vector4 GetPixel(int x, int y, int z)
            {
                return _rawTextureData[z][x + y * _width];
            }
            
            public Vector4 GetPixelTrilinear(float u, float v, float w)
            {
                int x1 = Mathf.FloorToInt(u * (_width - 1));
                int x2 = Mathf.CeilToInt(u * (_width - 1));
                int y1 = Mathf.FloorToInt(v * (_height - 1));
                int y2 = Mathf.CeilToInt(v * (_height - 1));
                int z1 = Mathf.FloorToInt(w * (_depth - 1));
                int z2 = Mathf.CeilToInt(w * (_depth - 1));

                
                Vector4 _111 = GetPixel(x1, y1, z1);
                Vector4 _121 = GetPixel(x1, y2, z1);
                Vector4 _211 = GetPixel(x2, y1, z1);
                Vector4 _221 = GetPixel(x2, y2, z1);

                Vector4 _111_121 = Vector4.Lerp(_111, _121, u - x1);
                Vector4 _211_221 = Vector4.Lerp(_211, _221, u - x1);
                
                Vector4 _z1s = Vector4.Lerp(_111_121, _211_221, v - y1);


                Vector4 _112 = GetPixel(x1, y1, z2);
                Vector4 _122 = GetPixel(x1, y2, z2);
                Vector4 _212 = GetPixel(x2, y1, z2);
                Vector4 _222 = GetPixel(x2, y2, z2);
                
                Vector4 _112_122 = Vector4.Lerp(_112, _122, u - x1);
                Vector4 _212_222 = Vector4.Lerp(_212, _222, u - x1);
                
                Vector4 _z2s = Vector4.Lerp(_112_122, _212_222, v - y1);
                

                return Vector4.Lerp(_z1s, _z2s, w - z1);
            }
        }
        
        /// <summary>
        /// Allocates a read/write render texture for linear (non-color) floating point data
        /// with given width, height, and component count,
        /// where component count dictates # of floats allocated per pixel.
        /// </summary>
        public static RenderTexture AllocateRWLinearRT(int width, int height, int depth = 0, int componentCount = 4, TextureWrapMode wrapMode = TextureWrapMode.Clamp, FilterMode filterMode = FilterMode.Bilinear)
        {
            // Determine format from component count
            RenderTextureFormat format;
            switch (componentCount)
            {
                case 1:
                    format = RenderTextureFormat.RFloat;
                    break;
                case 2:
                    format = RenderTextureFormat.RGFloat;
                    break;
                default:  // Unfortunately no RGBFloat, so default to float4 ARGBFloat for both 3 and 4 component counts
                    format = RenderTextureFormat.ARGBFloat;
                    break;
            }

            // Construct the render texture using given parameters
            RenderTexture texture = new RenderTexture(width, height, 0, format, RenderTextureReadWrite.Linear);
            // If 3D render texture
            if (depth > 0)
            {
                texture.dimension = UnityEngine.Rendering.TextureDimension.Tex3D;
                texture.volumeDepth = depth;
            }
            texture.wrapMode = wrapMode;
            texture.filterMode = filterMode;
            texture.enableRandomWrite = true;
            texture.Create();

            return texture;
        }
    }
}