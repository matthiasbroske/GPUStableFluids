using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace StableFluids
{
    /// <summary>
    /// A GPU implementation of Jos Stam's Stable Fluids in 3D.
    /// </summary>
    [RequireComponent(typeof(MeshRenderer))]
    public class StableFluids3D : MonoBehaviour
    {
        [Header("Dependencies")]
        [SerializeField] private ComputeShader _stableFluids3DCompute;
        [SerializeField] private Material _volumeMaterial;

        [Header("Fluid Simulation Parameters")] 
        [SerializeField] private Vector3Int _resolution;
        [SerializeField] private float _diffusion;

        [Header("Testing Parameters")]
        [SerializeField] private float _addVelocity;
        [SerializeField] private float _addDensity;
        [SerializeField] private float _addVelocityRadius;
        [SerializeField] private float _addDensityRadius;

        // Input
        private Vector2 _previousMousePosition;
        
        // Mesh renderer
        private MeshRenderer _meshRenderer;

        // Render textures
        private RenderTexture _densityOutTexture;
        private RenderTexture _densityInTexture;
        private RenderTexture _velocityOutTexture;
        private RenderTexture _velocityInTexture;
        private RenderTexture _pressureOutTexture;
        private RenderTexture _pressureInTexture;
        private RenderTexture _divergenceTexture;
        
        // Compute kernels
        private int _clearKernel;
        private int _addValueKernel;
        private int _advectionKernel;
        private int _diffusionKernel;
        private int _projectionPt1Kernel;
        private int _projectionPt2Kernel;
        private int _projectionPt3Kernel;

        // Thread counts
        private Vector3Int _threadCounts;

        private void Start()
        {
            // Setup mesh renderer
            _meshRenderer = GetComponent<MeshRenderer>();
            _meshRenderer.material = _volumeMaterial;

            // Get kernel ids
            _clearKernel = _stableFluids3DCompute.FindKernel("Clear");
            _addValueKernel = _stableFluids3DCompute.FindKernel("AddValue");
            _advectionKernel = _stableFluids3DCompute.FindKernel("Advection");
            _diffusionKernel = _stableFluids3DCompute.FindKernel("Diffusion");
            _projectionPt1Kernel = _stableFluids3DCompute.FindKernel("ProjectionPt1");
            _projectionPt2Kernel = _stableFluids3DCompute.FindKernel("ProjectionPt2");
            _projectionPt3Kernel = _stableFluids3DCompute.FindKernel("ProjectionPt3");
            
            // Get thread counts
            _stableFluids3DCompute.GetKernelThreadGroupSizes(_advectionKernel, 
                out uint xThreadGroupSize, out uint yThreadGroupSize, out uint zThreadGroupSize);
            _threadCounts = new Vector3Int(
                Mathf.CeilToInt(_resolution.x / (float)xThreadGroupSize),
                Mathf.CeilToInt(_resolution.y / (float)yThreadGroupSize),
                Mathf.CeilToInt(_resolution.z / (float)zThreadGroupSize));

            // Pass constants to compute
            _stableFluids3DCompute.SetInts("_Resolution", new int[] { _resolution.x, _resolution.y, _resolution.z });

            // Allocate render textures
            _densityOutTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, _resolution.z, 3);
            _densityInTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, _resolution.z, 3);
            _velocityOutTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, _resolution.z, 3);
            _velocityInTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, _resolution.z, 3);
            _pressureOutTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, _resolution.z, 1);
            _pressureInTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, _resolution.z, 1);
            _divergenceTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, _resolution.z, 1);
            
            // Pass params to volume rendering material
            _volumeMaterial.SetTexture("_Voxels", _densityOutTexture);
            _volumeMaterial.SetVector("_BoundsMin", _meshRenderer.bounds.min);
            _volumeMaterial.SetVector("_BoundsMax", _meshRenderer.bounds.max);
            _volumeMaterial.SetVector("_Dimensions", (Vector3)_resolution);
            
            // Reset simulation
            Reset();
        }

        /// <summary>
        /// Process input and run fluid simulation.
        /// </summary>
        void Update()
        {
            // Input handling
            // Vector2 mousePosition = Input.mousePosition;
            // Vector2 scaledMousePosition = mousePosition / new Vector2(Screen.width, Screen.height) * (Vector2) _resolution;
            // if (Input.GetMouseButton(1))
            // {
            //     Color density = Color.HSVToRGB(Mathf.Repeat(Time.time*0.2f, 1), 0.8f, 0.4f) * _addDensity * Time.deltaTime;
                 AddValueToTexture(_densityOutTexture, _resolution/2, Vector4.one * Time.deltaTime * _addDensity, _addDensityRadius);
            // }
            // if (Input.GetMouseButton(0))
            // {
            //     Vector2 mouseVelocity = (mousePosition - _previousMousePosition) * _addVelocity * Time.deltaTime;
                 AddValueToTexture(_velocityOutTexture, _resolution/2, Vector4.one * Time.deltaTime * _addVelocity, _addVelocityRadius);
            // }
            // if (Input.GetKeyDown(KeyCode.R))
            // {
            //     Reset();
            // }
            // _previousMousePosition = mousePosition;

            // Update compute parameters
            float alpha = Time.deltaTime * _diffusion * (_resolution.x - 2) * (_resolution.y - 2) * (_resolution.z - 2);
            float beta = 1 / (1 + 4 * alpha);
            _stableFluids3DCompute.SetFloat("_Alpha", alpha);
            _stableFluids3DCompute.SetFloat("_Beta", beta);
            _stableFluids3DCompute.SetFloat("_DeltaTime", Time.deltaTime);

            // Run the fluid simulation
            UpdateVelocity();
            UpdateDensity();
        }

        /// <summary>
        /// Add value around given position in target texture.
        /// </summary>
        private void AddValueToTexture(RenderTexture target, Vector3 position, Vector4 value, float radius)
        {
            _stableFluids3DCompute.SetFloat("_AddRadius", radius);
            _stableFluids3DCompute.SetVector("_AddPosition", position);
            _stableFluids3DCompute.SetVector("_AddValue", value);
            _stableFluids3DCompute.SetTexture(_addValueKernel, "_XOut", target);
            _stableFluids3DCompute.Dispatch(_addValueKernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
        }

        /// <summary>
        /// Diffuse and advect density.
        /// </summary>
        private void UpdateDensity()
        {
            Diffuse(_densityInTexture, _densityOutTexture);
            Graphics.CopyTexture(_densityOutTexture, _densityInTexture);  // Diffuse output is advect input
            Advect(_densityInTexture, _densityOutTexture);
        }
        
        /// <summary>
        /// Diffuse and advect velocity, maintaining fluid incompressibility.
        /// </summary>
        private void UpdateVelocity()
        {
            Diffuse(_velocityInTexture, _velocityOutTexture);
            Project();
            Graphics.CopyTexture(_velocityOutTexture, _velocityInTexture); // Diffuse output is advect input
            Advect(_velocityInTexture, _velocityOutTexture);
            Project();
        }

        /// <summary>
        /// Diffuse input texture using Gauss-Seidel.
        /// </summary>
        /// <param name="inTexture"></param>
        /// <param name="outTexture"></param>
        private void Diffuse(RenderTexture inTexture, RenderTexture outTexture)
        {
            for (int k = 0; k < 10; k++)
            {
                _stableFluids3DCompute.SetTexture(_diffusionKernel, "_XIn", outTexture);
                _stableFluids3DCompute.SetTexture(_diffusionKernel, "_XOut", inTexture);
                _stableFluids3DCompute.Dispatch(_diffusionKernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
                _stableFluids3DCompute.SetTexture(_diffusionKernel, "_XIn", inTexture);
                _stableFluids3DCompute.SetTexture(_diffusionKernel, "_XOut", outTexture);
                _stableFluids3DCompute.Dispatch(_diffusionKernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
            }
        }

        /// <summary>
        /// Advect input texture by velocity.
        /// </summary>
        private void Advect(RenderTexture inTexture, RenderTexture outTexture)
        {
            _stableFluids3DCompute.SetTexture(_advectionKernel, "_Velocity", _velocityInTexture);
            _stableFluids3DCompute.SetTexture(_advectionKernel, "_XIn", inTexture);
            _stableFluids3DCompute.SetTexture(_advectionKernel, "_XOut", outTexture);
            _stableFluids3DCompute.Dispatch(_advectionKernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
        }

        /// <summary>
        /// Correct the velocities such that pressure is zero.
        /// </summary>
        private void Project()
        {
            // Projection Part 1
            _stableFluids3DCompute.SetTexture(_projectionPt1Kernel, "_PressureOut", _pressureOutTexture);
            _stableFluids3DCompute.SetTexture(_projectionPt1Kernel, "_Divergence", _divergenceTexture);
            _stableFluids3DCompute.SetTexture(_projectionPt1Kernel, "_Velocity", _velocityOutTexture);
            _stableFluids3DCompute.Dispatch(_projectionPt1Kernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);

            // Projection Pt2
            for (int k = 0; k < 10; k++)
            {
                _stableFluids3DCompute.SetTexture(_projectionPt2Kernel, "_Divergence", _divergenceTexture);
                _stableFluids3DCompute.SetTexture(_projectionPt2Kernel, "_PressureIn", _pressureOutTexture);
                _stableFluids3DCompute.SetTexture(_projectionPt2Kernel, "_PressureOut", _pressureInTexture);
                _stableFluids3DCompute.Dispatch(_projectionPt2Kernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
                _stableFluids3DCompute.SetTexture(_projectionPt2Kernel, "_PressureIn", _pressureInTexture);
                _stableFluids3DCompute.SetTexture(_projectionPt2Kernel, "_PressureOut", _pressureOutTexture);
                _stableFluids3DCompute.Dispatch(_projectionPt2Kernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
            }

            // Projection Pt3
            _stableFluids3DCompute.SetTexture(_projectionPt3Kernel, "_Velocity", _velocityOutTexture);
            _stableFluids3DCompute.SetTexture(_projectionPt3Kernel, "_PressureIn", _pressureOutTexture);
            _stableFluids3DCompute.Dispatch(_projectionPt3Kernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
        }

        /// <summary>
        /// Reset the simulation.
        /// </summary>
        private void Reset()
        {
            // Clear density and velocity textures
            _stableFluids3DCompute.SetTexture(_clearKernel,"_XOut", _densityInTexture);
            _stableFluids3DCompute.Dispatch(_clearKernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
            _stableFluids3DCompute.SetTexture(_clearKernel,"_XOut", _densityInTexture);
            _stableFluids3DCompute.Dispatch(_clearKernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
            _stableFluids3DCompute.SetTexture(_clearKernel,"_XOut", _velocityInTexture);
            _stableFluids3DCompute.Dispatch(_clearKernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
            _stableFluids3DCompute.SetTexture(_clearKernel,"_XOut", _velocityOutTexture);
            _stableFluids3DCompute.Dispatch(_clearKernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
        }
    }
}
