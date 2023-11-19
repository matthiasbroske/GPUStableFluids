using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace StableFluids
{
    /// <summary>
    /// A GPU implementation of Jos Stam's Stable Fluids in 2D.
    /// </summary>
    public class StableFluids2D : MonoBehaviour
    {
        [Header("Dependencies")]
        [SerializeField] private ComputeShader _stableFluids2DCompute;

        [Header("Fluid Simulation Parameters")] 
        [SerializeField] private Vector2Int _resolution;
        [SerializeField] private float _diffusion;

        [Header("Testing Parameters")] 
        [SerializeField] private Texture2D _initialImage;
        [SerializeField] private float _addVelocity;
        [SerializeField] private float _addDensity;
        [SerializeField] private float _addVelocityRadius;
        [SerializeField] private float _addDensityRadius;

        // Input
        private Vector2 _previousMousePosition;

        // Render textures
        private RenderTexture _displayTexture;
        private RenderTexture _densityOutTexture;
        private RenderTexture _densityInTexture;
        private RenderTexture _velocityOutTexture;
        private RenderTexture _velocityInTexture;
        private RenderTexture _pressureOutTexture;
        private RenderTexture _pressureInTexture;
        private RenderTexture _divergenceTexture;
        
        // Compute kernels
        private int _addValueKernel;
        private int _advectionKernel;
        private int _diffusionKernel;
        private int _projectionPt1Kernel;
        private int _projectionPt2Kernel;
        private int _projectionPt3Kernel;
        private int _setBoundsXKernel;
        private int _setBoundsYKernel;

        // Thread counts
        private Vector3Int _threadCounts;
        private int _setBoundsXThreadCount;
        private int _setBoundsYThreadCount;

        private void Start()
        {
            // Get kernel ids
            _addValueKernel = _stableFluids2DCompute.FindKernel("AddValue");
            _advectionKernel = _stableFluids2DCompute.FindKernel("Advection");
            _diffusionKernel = _stableFluids2DCompute.FindKernel("Diffusion");
            _projectionPt1Kernel = _stableFluids2DCompute.FindKernel("ProjectionPt1");
            _projectionPt2Kernel = _stableFluids2DCompute.FindKernel("ProjectionPt2");
            _projectionPt3Kernel = _stableFluids2DCompute.FindKernel("ProjectionPt3");
            _setBoundsXKernel = _stableFluids2DCompute.FindKernel("SetBoundsX");
            _setBoundsYKernel = _stableFluids2DCompute.FindKernel("SetBoundsY");
            
            // Get thread counts
            _stableFluids2DCompute.GetKernelThreadGroupSizes(_advectionKernel, 
                out uint xThreadGroupSize, out uint yThreadGroupSize, out uint zThreadGroupSize);
            _threadCounts = new Vector3Int(
                Mathf.CeilToInt(_resolution.x / (float)xThreadGroupSize),
                Mathf.CeilToInt(_resolution.y / (float)yThreadGroupSize),
                Mathf.CeilToInt(1 / (float)zThreadGroupSize));
            _stableFluids2DCompute.GetKernelThreadGroupSizes(_setBoundsXKernel, 
                out xThreadGroupSize, out yThreadGroupSize, out zThreadGroupSize);
            _setBoundsXThreadCount = Mathf.CeilToInt(_resolution.x / (float)xThreadGroupSize);
            _stableFluids2DCompute.GetKernelThreadGroupSizes(_setBoundsYKernel, 
                out xThreadGroupSize, out yThreadGroupSize, out zThreadGroupSize);
            _setBoundsYThreadCount = Mathf.CeilToInt(_resolution.y / (float)xThreadGroupSize);

            // Pass constants to compute
            _stableFluids2DCompute.SetInts("_Resolution", new int[] { _resolution.x, _resolution.y });

            // Subscribe to end of render pipeline event
            RenderPipelineManager.endContextRendering += OnEndContextRendering;
            
            // Allocate render textures
            _displayTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, 0, 4);
            _densityOutTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, 0, 3);
            _densityInTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, 0, 3);
            _velocityOutTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, 0, 2);
            _velocityInTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, 0, 2);
            _pressureOutTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, 0, 1);
            _pressureInTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, 0, 1);
            _divergenceTexture = RenderTextureUtilities.AllocateRWLinearRT(_resolution.x, _resolution.y, 0, 1);

            // Reset simulation
            Reset();
        }

        /// <summary>
        /// Process input and run fluid simulation.
        /// </summary>
        void Update()
        {
            // Input handling
            Vector2 mousePosition = Input.mousePosition;
            Vector2 scaledMousePosition = mousePosition / new Vector2(Screen.width, Screen.height) * _resolution;
            if (Input.GetMouseButton(1))
            {
                Color density = Color.HSVToRGB(Mathf.Repeat(Time.time*0.2f, 1), 0.8f, 0.4f) * _addDensity * Time.deltaTime;
                AddValueToTexture(_densityOutTexture, scaledMousePosition, density, _addDensityRadius);
            }
            if (Input.GetMouseButton(0))
            {
                Vector2 mouseVelocity = (mousePosition - _previousMousePosition) * _addVelocity * Time.deltaTime;
                AddValueToTexture(_velocityOutTexture, scaledMousePosition, mouseVelocity, _addVelocityRadius);
            }
            if (Input.GetKeyDown(KeyCode.R))
            {
                Reset();
            }
            _previousMousePosition = mousePosition;

            // Update compute parameters
            float alpha = Time.deltaTime * _diffusion * (_resolution.x - 2) * (_resolution.y - 2);
            float beta = 1 / (1 + 4 * alpha);
            _stableFluids2DCompute.SetFloat("_Alpha", alpha);
            _stableFluids2DCompute.SetFloat("_Beta", beta);
            _stableFluids2DCompute.SetFloat("_DeltaTime", Time.deltaTime);

            // Run the fluid simulation
            UpdateVelocity();
            UpdateDensity();
            
            // Copy density to display
            Graphics.CopyTexture(_densityOutTexture, _displayTexture);
        }

        /// <summary>
        /// Add value around given position in target texture.
        /// </summary>
        private void AddValueToTexture(RenderTexture target, Vector2 position, Vector4 value, float radius)
        {
            _stableFluids2DCompute.SetFloat("_AddRadius", radius);
            _stableFluids2DCompute.SetFloats("_AddPosition", new float[2] { position.x, position.y });
            _stableFluids2DCompute.SetVector("_AddValue", value);
            _stableFluids2DCompute.SetTexture(_addValueKernel, "_XOut", target);
            _stableFluids2DCompute.Dispatch(_addValueKernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
        }

        /// <summary>
        /// Diffuse and advect density.
        /// </summary>
        private void UpdateDensity()
        {
            Diffuse(_densityInTexture, _densityOutTexture);
            Graphics.CopyTexture(_densityOutTexture, _densityInTexture);  // Swap output to input
            Advect(_densityInTexture, _densityOutTexture);
        }
        
        /// <summary>
        /// Diffuse and advect velocity, maintaining fluid incompressibility.
        /// </summary>
        private void UpdateVelocity()
        {
            Diffuse(_velocityInTexture, _velocityOutTexture, true);
            ProjectVelocity();
            Graphics.CopyTexture(_velocityOutTexture, _velocityInTexture); // Swap output to input
            Advect(_velocityInTexture, _velocityOutTexture, true);
            ProjectVelocity();
        }

        /// <summary>
        /// Diffuse input texture using Gauss-Seidel.
        /// </summary>
        /// <param name="inTexture"></param>
        /// <param name="outTexture"></param>
        private void Diffuse(RenderTexture inTexture, RenderTexture outTexture, bool setBounds = false)
        {
            for (int k = 0; k < 10; k++)
            {
                _stableFluids2DCompute.SetTexture(_diffusionKernel, "_XIn", outTexture);
                _stableFluids2DCompute.SetTexture(_diffusionKernel, "_XOut", inTexture);
                _stableFluids2DCompute.Dispatch(_diffusionKernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
                _stableFluids2DCompute.SetTexture(_diffusionKernel, "_XIn", inTexture);
                _stableFluids2DCompute.SetTexture(_diffusionKernel, "_XOut", outTexture);
                _stableFluids2DCompute.Dispatch(_diffusionKernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
                if (setBounds) SetBounds(inTexture, outTexture);
            }
        }

        /// <summary>
        /// Advect input texture by velocity.
        /// </summary>
        private void Advect(RenderTexture inTexture, RenderTexture outTexture, bool setBounds = false)
        {
            _stableFluids2DCompute.SetTexture(_advectionKernel, "_Velocity", _velocityInTexture);
            _stableFluids2DCompute.SetTexture(_advectionKernel, "_XIn", inTexture);
            _stableFluids2DCompute.SetTexture(_advectionKernel, "_XOut", outTexture);
            _stableFluids2DCompute.Dispatch(_advectionKernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
            if (setBounds) SetBounds(inTexture, outTexture);
        }

        /// <summary>
        /// Correct the velocities such that pressure is zero.
        /// </summary>
        private void ProjectVelocity()
        {
            // Projection Part 1
            _stableFluids2DCompute.SetTexture(_projectionPt1Kernel, "_PressureOut", _pressureOutTexture);
            _stableFluids2DCompute.SetTexture(_projectionPt1Kernel, "_Divergence", _divergenceTexture);
            _stableFluids2DCompute.SetTexture(_projectionPt1Kernel, "_Velocity", _velocityOutTexture);
            _stableFluids2DCompute.Dispatch(_projectionPt1Kernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
            SetBounds(_velocityInTexture, _velocityOutTexture);
            
            // Projection Pt2
            for (int k = 0; k < 10; k++)
            {
                _stableFluids2DCompute.SetTexture(_projectionPt2Kernel, "_Divergence", _divergenceTexture);
                _stableFluids2DCompute.SetTexture(_projectionPt2Kernel, "_PressureIn", _pressureOutTexture);
                _stableFluids2DCompute.SetTexture(_projectionPt2Kernel, "_PressureOut", _pressureInTexture);
                _stableFluids2DCompute.Dispatch(_projectionPt2Kernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
                _stableFluids2DCompute.SetTexture(_projectionPt2Kernel, "_PressureIn", _pressureInTexture);
                _stableFluids2DCompute.SetTexture(_projectionPt2Kernel, "_PressureOut", _pressureOutTexture);
                _stableFluids2DCompute.Dispatch(_projectionPt2Kernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
                SetBounds(_velocityInTexture, _velocityOutTexture);
            }

            // Projection Pt3
            _stableFluids2DCompute.SetTexture(_projectionPt3Kernel, "_Velocity", _velocityOutTexture);
            _stableFluids2DCompute.SetTexture(_projectionPt3Kernel, "_PressureIn", _pressureOutTexture);
            _stableFluids2DCompute.Dispatch(_projectionPt3Kernel, _threadCounts.x, _threadCounts.y, _threadCounts.z);
            SetBounds(_velocityInTexture, _velocityOutTexture);
        }
        
        /// <summary>
        /// Enforces boundary condition around edges.
        /// </summary>
        private void SetBounds(RenderTexture inTexture, RenderTexture outTexture)
        { 
            _stableFluids2DCompute.SetTexture(_setBoundsXKernel, "_XIn", outTexture);
            _stableFluids2DCompute.SetTexture(_setBoundsXKernel, "_XOut", inTexture);
            _stableFluids2DCompute.Dispatch(_setBoundsXKernel, _setBoundsXThreadCount, 1, 1);
            _stableFluids2DCompute.SetTexture(_setBoundsYKernel, "_XIn", inTexture);
            _stableFluids2DCompute.SetTexture(_setBoundsYKernel, "_XOut", outTexture);
            _stableFluids2DCompute.Dispatch(_setBoundsYKernel, _setBoundsYThreadCount, 1, 1);
        }

        /// <summary>
        /// Reset the simulation.
        /// </summary>
        private void Reset()
        {
            // Initialize "density" textures with initial image
            Graphics.Blit(_initialImage, _densityInTexture);
            Graphics.Blit(_initialImage, _densityOutTexture);
            
            // Initialize velocity to zero
            Graphics.Blit(Texture2D.blackTexture, _velocityInTexture);
            Graphics.Blit(Texture2D.blackTexture, _velocityOutTexture);
        }

        /// <summary>
        /// Blit display texture to screen at end of render pipeline.
        /// </summary>
        void OnEndContextRendering(ScriptableRenderContext context, List<Camera> cameras)
        {
            foreach (Camera cam in cameras)
            {
                Graphics.Blit(_displayTexture, cam.activeTexture);
            }
        }

        private void OnDestroy()
        {
            RenderPipelineManager.endContextRendering -= OnEndContextRendering;
        }
    }
}
