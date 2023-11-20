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
        [SerializeField] private float _addDensity;
        [SerializeField] private float _addDensityRadius;
        [SerializeField] private float _velocityMagnitude;
        [SerializeField] private float _addVelocityRadius;
        [SerializeField] private float _velocityFluctuation;
        [SerializeField] private float _velocityFluctuationRate;
        [SerializeField] private float _phiMagnitude;
        [SerializeField] private float _thetaMagnitude;
        [SerializeField] private float _thetaRate;
        [SerializeField] private float _phiRate;

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
        private int _setBoundsXYKernel;
        private int _setBoundsYZKernel;
        private int _setBoundsZXKernel;

        // Thread groups
        private Vector3Int _threadGroups;
        private Vector3Int _setBoundsXYThreadGroups;
        private Vector3Int _setBoundsYZThreadGroups;
        private Vector3Int _setBoundsZXThreadGroups;

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
            _setBoundsXYKernel = _stableFluids3DCompute.FindKernel("SetBoundsXY");
            _setBoundsYZKernel = _stableFluids3DCompute.FindKernel("SetBoundsYZ");
            _setBoundsZXKernel = _stableFluids3DCompute.FindKernel("SetBoundsZX");
            
            // Get thread counts
            _stableFluids3DCompute.GetKernelThreadGroupSizes(_advectionKernel, out uint xThreadGroupSize, out uint yThreadGroupSize, out uint zThreadGroupSize);
            _threadGroups = new Vector3Int(
                Mathf.CeilToInt(_resolution.x / (float)xThreadGroupSize),
                Mathf.CeilToInt(_resolution.y / (float)yThreadGroupSize),
                Mathf.CeilToInt(_resolution.z / (float)zThreadGroupSize));
            _stableFluids3DCompute.GetKernelThreadGroupSizes(_setBoundsXYKernel, out xThreadGroupSize, out yThreadGroupSize, out zThreadGroupSize);
            _setBoundsXYThreadGroups = new Vector3Int(
                Mathf.CeilToInt(_resolution.x * 2 / (float)xThreadGroupSize),
                Mathf.CeilToInt(_resolution.y / (float)yThreadGroupSize),
                Mathf.CeilToInt(1 / (float)zThreadGroupSize));
            _stableFluids3DCompute.GetKernelThreadGroupSizes(_setBoundsYZKernel, out xThreadGroupSize, out yThreadGroupSize, out zThreadGroupSize);
            _setBoundsYZThreadGroups = new Vector3Int(
                Mathf.CeilToInt(1 / (float)xThreadGroupSize),
                Mathf.CeilToInt(_resolution.y * 2/ (float)yThreadGroupSize),
                Mathf.CeilToInt(_resolution.z / (float)zThreadGroupSize));
            _stableFluids3DCompute.GetKernelThreadGroupSizes(_setBoundsZXKernel, out xThreadGroupSize, out yThreadGroupSize, out zThreadGroupSize);
            _setBoundsZXThreadGroups = new Vector3Int(
                Mathf.CeilToInt(_resolution.x / (float)xThreadGroupSize),
                Mathf.CeilToInt(1 / (float)yThreadGroupSize),
                Mathf.CeilToInt(_resolution.z * 2 / (float)zThreadGroupSize));

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
            _volumeMaterial.SetVector("_Dimensions", (Vector3)_resolution);
            
            // Reset simulation
            Reset();
        }

        /// <summary>
        /// Process input and run fluid simulation.
        /// </summary>
        void Update()
        {
            // Input
            if (Input.GetKeyDown(KeyCode.R))
            {
                Reset();
            }
            
            // Add density to center
            AddValueToTexture(_densityOutTexture, _resolution/2, Vector4.one * Time.deltaTime * _addDensity, _addDensityRadius);

            // Add velocity in semi-random oscillating direction
            float theta = Mathf.Sin(Time.time * _thetaRate) * _thetaMagnitude;
            float phi = Mathf.Cos(Time.time * _phiRate) * _phiMagnitude;
            float magnitude = _velocityMagnitude + Mathf.Sin(Time.time * _velocityFluctuationRate) * _velocityFluctuation;
            Vector3 velocity = new Vector3(
                Mathf.Cos(theta) * Mathf.Sin(phi),
                Mathf.Sin(theta) * Mathf.Sin(phi),
                Mathf.Cos(phi)
            ) * magnitude;
            AddValueToTexture(_velocityOutTexture, _resolution/2, velocity * Time.deltaTime, _addVelocityRadius);

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
            _stableFluids3DCompute.Dispatch(_addValueKernel, _threadGroups.x, _threadGroups.y, _threadGroups.z);
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
            Diffuse(_velocityInTexture, _velocityOutTexture, true);
            Project();
            Graphics.CopyTexture(_velocityOutTexture, _velocityInTexture); // Diffuse output is advect input
            Advect(_velocityInTexture, _velocityOutTexture, true);
            Project();
        }

        /// <summary>
        /// Diffuse input texture using Gauss-Seidel.
        /// </summary>
        private void Diffuse(RenderTexture inTexture, RenderTexture outTexture, bool setBounds = false)
        {
            for (int k = 0; k < 10; k++)
            {
                _stableFluids3DCompute.SetTexture(_diffusionKernel, "_XIn", outTexture);
                _stableFluids3DCompute.SetTexture(_diffusionKernel, "_XOut", inTexture);
                _stableFluids3DCompute.Dispatch(_diffusionKernel, _threadGroups.x, _threadGroups.y, _threadGroups.z);
                _stableFluids3DCompute.SetTexture(_diffusionKernel, "_XIn", inTexture);
                _stableFluids3DCompute.SetTexture(_diffusionKernel, "_XOut", outTexture);
                _stableFluids3DCompute.Dispatch(_diffusionKernel, _threadGroups.x, _threadGroups.y, _threadGroups.z);
                if (setBounds) SetBounds(inTexture, outTexture);
            }
        }

        /// <summary>
        /// Advect input texture by velocity.
        /// </summary>
        private void Advect(RenderTexture inTexture, RenderTexture outTexture, bool setBounds = false)
        {
            _stableFluids3DCompute.SetTexture(_advectionKernel, "_Velocity", _velocityInTexture);
            _stableFluids3DCompute.SetTexture(_advectionKernel, "_XIn", inTexture);
            _stableFluids3DCompute.SetTexture(_advectionKernel, "_XOut", outTexture);
            _stableFluids3DCompute.Dispatch(_advectionKernel, _threadGroups.x, _threadGroups.y, _threadGroups.z);
            if (setBounds) SetBounds(inTexture, outTexture);
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
            _stableFluids3DCompute.Dispatch(_projectionPt1Kernel, _threadGroups.x, _threadGroups.y, _threadGroups.z);
            SetBounds(_velocityInTexture, _velocityOutTexture);
            
            // Projection Pt2
            for (int k = 0; k < 10; k++)
            {
                _stableFluids3DCompute.SetTexture(_projectionPt2Kernel, "_Divergence", _divergenceTexture);
                _stableFluids3DCompute.SetTexture(_projectionPt2Kernel, "_PressureIn", _pressureOutTexture);
                _stableFluids3DCompute.SetTexture(_projectionPt2Kernel, "_PressureOut", _pressureInTexture);
                _stableFluids3DCompute.Dispatch(_projectionPt2Kernel, _threadGroups.x, _threadGroups.y, _threadGroups.z);
                _stableFluids3DCompute.SetTexture(_projectionPt2Kernel, "_PressureIn", _pressureInTexture);
                _stableFluids3DCompute.SetTexture(_projectionPt2Kernel, "_PressureOut", _pressureOutTexture);
                _stableFluids3DCompute.Dispatch(_projectionPt2Kernel, _threadGroups.x, _threadGroups.y, _threadGroups.z);
            }

            // Projection Pt3
            _stableFluids3DCompute.SetTexture(_projectionPt3Kernel, "_Velocity", _velocityOutTexture);
            _stableFluids3DCompute.SetTexture(_projectionPt3Kernel, "_PressureIn", _pressureOutTexture);
            _stableFluids3DCompute.Dispatch(_projectionPt3Kernel, _threadGroups.x, _threadGroups.y, _threadGroups.z);
            SetBounds(_velocityInTexture, _velocityOutTexture);
        }
        
        /// <summary>
        /// Enforces boundary condition around faces.
        /// </summary>
        private void SetBounds(RenderTexture inTexture, RenderTexture outTexture)
        { 
            _stableFluids3DCompute.SetTexture(_setBoundsXYKernel, "_XIn", outTexture);
            _stableFluids3DCompute.SetTexture(_setBoundsXYKernel, "_XOut", inTexture);
            _stableFluids3DCompute.Dispatch(_setBoundsXYKernel, _setBoundsXYThreadGroups.x, _setBoundsXYThreadGroups.y, _setBoundsXYThreadGroups.z);
            _stableFluids3DCompute.SetTexture(_setBoundsXYKernel, "_XIn", inTexture);
            _stableFluids3DCompute.SetTexture(_setBoundsXYKernel, "_XOut", outTexture);
            _stableFluids3DCompute.Dispatch(_setBoundsXYKernel, _setBoundsYZThreadGroups.x, _setBoundsYZThreadGroups.y, _setBoundsYZThreadGroups.z);
            _stableFluids3DCompute.SetTexture(_setBoundsZXKernel, "_XIn", outTexture);
            _stableFluids3DCompute.SetTexture(_setBoundsZXKernel, "_XOut", inTexture);
            _stableFluids3DCompute.Dispatch(_setBoundsZXKernel, _setBoundsZXThreadGroups.x, _setBoundsZXThreadGroups.y, _setBoundsZXThreadGroups.z);
        }

        /// <summary>
        /// Reset the simulation.
        /// </summary>
        private void Reset()
        {
            // Clear density and velocity textures
            _stableFluids3DCompute.SetTexture(_clearKernel,"_XOut", _densityInTexture);
            _stableFluids3DCompute.Dispatch(_clearKernel, _threadGroups.x, _threadGroups.y, _threadGroups.z);
            _stableFluids3DCompute.SetTexture(_clearKernel,"_XOut", _densityOutTexture);
            _stableFluids3DCompute.Dispatch(_clearKernel, _threadGroups.x, _threadGroups.y, _threadGroups.z);
            _stableFluids3DCompute.SetTexture(_clearKernel,"_XOut", _velocityInTexture);
            _stableFluids3DCompute.Dispatch(_clearKernel, _threadGroups.x, _threadGroups.y, _threadGroups.z);
            _stableFluids3DCompute.SetTexture(_clearKernel,"_XOut", _velocityOutTexture);
            _stableFluids3DCompute.Dispatch(_clearKernel, _threadGroups.x, _threadGroups.y, _threadGroups.z);
        }
    }
}
