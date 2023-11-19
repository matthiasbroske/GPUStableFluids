Shader "Volume/Default"
{
	Properties
	{
		// __ Inspector visible inputs for debugging purposes _________________
		[MaterialToggle] _UseColorMap("Use Color Map", Float) = 1
		_ColorMap("Color Map Texture", 2D) = "" {}
		_Color("Color", Color) = (1, 1, 1, 1)
		[MaterialToggle] _UseOpacityMap("Use Opacity Map", Float) = 1
		_OpacityMap("Opacity Map Texture", 2D) = "" {}
		_Opacity("Opacity", Color) = (0.1, 0.1, 0.1, 1)
		_OpacityMultiplier("Opacity Multiplier", Float) = 0.1
		_ColorDataMin("Color Data Min", Float) = 0.0
		_ColorDataMax("Color Data Max", Float) = 1.0
		_Brightness("Brightness", Float) = 0.1
		_StepCount("Step Count", Int) = 200
        _BoundsMin("Bounds Min", Vector) = (0,0,0,0)
        _BoundsMax("Bounds Max", Vector) = (0,0,0,0)
        _Dimensions("Dimensions", Vector) = (0,0,0,0)
	}
		
	SubShader
	{
		Tags 
		{ 
			"RenderType" = "Transparent" 
			"Queue" = "Transparent+1" 
			"ForceNoShadowCasting" = "True"
		}

		Pass
		{
			// __ Shader Setup ________________________________________________
			ZWrite Off
			Cull Front
			ZTest Always
			Blend SrcAlpha OneMinusSrcAlpha

			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag

			#include "UnityCG.cginc"

			struct appdata
			{
				float4 vertex : POSITION;
				float2 uv : TEXCOORD0;
			};

			struct v2f
			{
				float2 uv : TEXCOORD0;
				float3 modelDirection : TEXCOORD1;
				float4 screenPosition : TEXCOORD2;
				float4 vertex : SV_POSITION;
				float3 modelSpaceCameraPos :TEXCOORD3;
				float4 modelPosition : TEXCOORD4;
				float3 viewPosition : TEXCOORD5;
			};

			// __ Uniforms ____________________________________________________
			sampler3D _Voxels;
			// Volume dimensions
            uint3 _Dimensions;
			// Bounds of the mesh the volume is being rendered on
			float3 _BoundsMin;
			float3 _BoundsMax;
			// Color/opacity map toggles
			int _UseColorMap;
			int _UseOpacityMap;
			// Textures for colormap (RGB) and opacitymap (grayscale) transfer functions
			sampler2D _ColorMap;
			sampler2D _OpacityMap;
			// Default color/opacity for when not using color/opacity maps
			float4 _Color;
			float4 _Opacity;
			// Range used to remap scalar data value to lookup value for color/opacity maps
			float _ColorDataMin;
			float _ColorDataMax;
			// Control over visual opacity of rendered volume
			float _OpacityMultiplier;
			// Intensity of ambient lighting
			float _Brightness;
			// Max number of steps traversed into the volume
			int _StepCount;
			// Depth texture for intersection with opaque objects
			// (written to automatically when camera's DepthTextureMode
			// is set to Depth)
			sampler2D_float _CameraDepthTexture;

			// __ Helper Functions ____________________________________________
			// Remaps dataValue from data range to target range
			//
			// dataValue = scalar value to be remapped
			// range = (dataMin, dataMax, targetRangeMin, targetRangeMax)
			float Remap(float dataValue, float4 range)
			{
				return range.z + (dataValue - range.x) * (range.w - range.z) / (range.y - range.x);
			}

			// Remaps vector from data range to 0-1
			float3 RemapVector01(float3 dataValue, float3 dataRange[2])
			{
				return float3(
					Remap(dataValue.x, float4(dataRange[0].x, dataRange[1].x, 0, 1)),
					Remap(dataValue.y, float4(dataRange[0].y, dataRange[1].y, 0, 1)),
					Remap(dataValue.z, float4(dataRange[0].z, dataRange[1].z, 0, 1))
				);
			}

			// Calculates minimum distance and maximum distance for a ray to
			// intersect a particular rectangular prism volume
			//
			// orig = ray origin (where camera is in model space)
			// invdir = inv direction the ray is coming from (camera origin to model)
			// sign = indicates direction in each axis
			// bounds = min/max of rectangular prism
			float2  Intersect(float3 orig, float3 invdir, int3 sign, float3 bounds[2]) {

				float tmin, tmax, tymin, tymax, tzmin, tzmax;
				tmin = (bounds[sign[0]].x - orig.x) * invdir.x;
				tmax = (bounds[1 - sign[0]].x - orig.x) * invdir.x;
				tymin = (bounds[sign[1]].y - orig.y) * invdir.y;
				tymax = (bounds[1 - sign[1]].y - orig.y) * invdir.y;

				if ((tmin > tymax) || (tymin > tmax))
					return float2(0,0);
				if (tymin > tmin)
					tmin = tymin;
				if (tymax < tmax)
					tmax = tymax;

				tzmin = (bounds[sign[2]].z - orig.z) * invdir.z;
				tzmax = (bounds[1 - sign[2]].z - orig.z) * invdir.z;

				if ((tmin > tzmax) || (tzmin > tmax))
					return float2(0,0);
				if (tzmin > tmin)
					tmin = tzmin;
				if (tzmax < tmax)
					tmax = tzmax;

				return float2(tmin,tmax);
			}


			// __ Vertex Shader _______________________________________________
			v2f vert(appdata v)
			{
				v2f o;

				// Typical boilerplate
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.uv = v.uv;

				// Save the clip space position so we can use it later
				o.screenPosition = ComputeScreenPos(o.vertex);
				o.modelPosition = v.vertex;

				// Save camera position in model/object space
				o.modelSpaceCameraPos = mul(unity_WorldToObject, float4(_WorldSpaceCameraPos, 1)).xyz;
				// Subtract camera position from vertex position 
				// to get a ray pointing from the camera to this vertex in model space
				o.modelDirection = v.vertex.xyz - o.modelSpaceCameraPos;

				// Save the view space position so we can use it in lighting calculations
				o.viewPosition = UnityObjectToViewPos(v.vertex.xyz);

				return o;
			}


			// __ Frag Shader _________________________________________________
			fixed4 frag(v2f i) : SV_Target
			{
				// Boilerplate perspective division of interpolated 2D screen position
				float perspectiveDivide = 1.0f / i.screenPosition.w;
				float3 direction = i.modelDirection * perspectiveDivide;
				float2 screenUV = (i.screenPosition.xy * perspectiveDivide);

				// Look up the first opaque object in the camera depth texture
				// this is in world-space-distance
				// (if this isn't working, double check the main camera's "Depth Texture" is set to on in the inspector)
				float obstacleDepth = LinearEyeDepth(UNITY_SAMPLE_DEPTH(tex2D(_CameraDepthTexture, screenUV)));

				// Create a bounding box that intersection tests can be run on
				float3 bounds[2] = { _BoundsMin, _BoundsMax };

				// Determine the depth in world space at which the current view ray intersects and exits the bounding box
				float2 minmax = Intersect(
					i.modelSpaceCameraPos,
					1.0 / direction,
					int3(direction.x < 0 ? 1 : 0, direction.y < 0 ? 1 : 0, direction.z < 0 ? 1 : 0),
					bounds
				);
				float frontDepth = minmax.x;
				float backDepth = minmax.y;

				// Only volume-render content between near clip and obstacle depth
				if (frontDepth < _ProjectionParams.y)
					frontDepth = _ProjectionParams.y;
				if (obstacleDepth < frontDepth)
					discard;
				if (obstacleDepth < backDepth)
					backDepth = obstacleDepth;

				// Calculate front/back intersection positions in model space 
				float3 frontCoord = direction * frontDepth + i.modelSpaceCameraPos;
				float3 backCoord = direction * backDepth + i.modelSpaceCameraPos;

				// Find the direction vector and distance between front and back ray intersection within volume
				float3 dist = backCoord - frontCoord;
				float len = length(dist);
				float3 dir = dist / len;

				// Destination color (used for blending accumulation)
				float4 dst = 0;
				// Final color to return
				fixed4 col = float4(0,1,0,1);

				// Length of the diagonal of the box
				float diagonal = length(_BoundsMax-_BoundsMin);
				// Distance traveled along the direction vector into the bounding box
				float traveled = 0;
				// Iteration step progress, caps iteration to max step count
				float progress = 0;

				// Jitter the start of each ray to avoid visual artifacts from uniform slices
				if (1 == 1) {
					float m = 1000;
					float a = 13489;
					float c = 21561;

					float m2 = 1000;
					float a2 = 67852;
					float c2 = 45932;

					uint Rx = (((a*uint(10000 * screenUV.x) + c) % (uint)(m)));
					uint Ry = (((a2*uint(10000 * screenUV.y) + c2) % (uint)(m2)));
					uint r = Rx ^ Ry;
					float R = r % 100 / 100.0;
					progress += R;
				}

				// Loop for step count (number of slices)
				for (; progress <= _StepCount; progress += 1) {

					///////////////////////////////////////////////////////////
					// Traverse the volume and read its scalar data
					///////////////////////////////////////////////////////////

					// Calculate the distance traveled into the bounding box
					// (0.9 -> decrease step size since rarely looking down full diagonal)
					traveled = progress / (float)_StepCount * diagonal * 0.9;

					// Update current position in the bounding box
					float3 pos = frontCoord + traveled * dir;

					// Remap the position to uvw-coords into volume texture
					float3 uvw = RemapVector01(pos, bounds);

					// Exit if full distance traversed
					if (traveled > len)
						break;

                    float voxel = tex3Dlod(_Voxels, float4(uvw, 0)).r;
					float dataValue = Remap(voxel, float4(_ColorDataMin, _ColorDataMax, 0, 1));

					///////////////////////////////////////////////////////////
					// Apply transfer function to determine color
					///////////////////////////////////////////////////////////

					// Look up the transfer function color and opacity
					float4 transferColor = _UseColorMap ? tex2Dlod(_ColorMap, clamp(dataValue, 0, 1)) : _Color;
					float4 transferOpacity = _UseOpacityMap ? tex2Dlod(_OpacityMap, clamp(dataValue, 0, 1)) : _Opacity;
					// Calculate single alpha value based on brightness of opacity map
					float alpha = clamp(transferOpacity.r * _OpacityMultiplier / _StepCount * 100, 0, 1);

					// Combine transfer color and opacity to get a full source color
					float4 src = float4(transferColor.rgb, alpha);

					///////////////////////////////////////////////////////////
					// Calculate lighting (performed in view/eye space)
					///////////////////////////////////////////////////////////

					// Ambient
					float3 litColor = _Brightness * src.rgb;

					// Apply
					// ========================================================
					src.rgb = litColor;

					///////////////////////////////////////////////////////////
					// Color blending
					///////////////////////////////////////////////////////////

					// Blend front to back
					src.rgb *= src.a; // pre-multiply alpha
					dst = (1.0f - dst.a) * src + dst; // accumulate

					if (dst.a >= 0.99)
						break;
				}

				// Build the final color
				col.a = clamp(dst.a, 0, 1);
				col.rgb = clamp(dst.rgb / dst.a, 0, 1); // un-pre-multiply alpha
				return col;
			}
			ENDCG
		}
	}
}