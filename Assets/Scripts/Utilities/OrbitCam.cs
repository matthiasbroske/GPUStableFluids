using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace StableFluids
{
    public class OrbitCam : MonoBehaviour
    {
        [Header("Settings")] 
        [SerializeField] private float _orbitRadius;
        [SerializeField] private float _horizontalOrbitSpeed;
        [SerializeField] private float _verticalOrbitSpeed;
        [SerializeField] private float _zoomSpeed;
        [SerializeField] private float _orbitX;
        [SerializeField] private float _orbitY;

        private void Update()
        {
            // Update orbit angles based on mouse input
            if (Input.GetMouseButton(0))
            {
                _orbitX += Input.GetAxis("Mouse Y") * _horizontalOrbitSpeed;
                _orbitY += Input.GetAxis("Mouse X") * _verticalOrbitSpeed;
            }
            _orbitX = Mathf.Clamp(_orbitX, 5, 85);
            
            // Update radius based on click and drag or mouse scroll
            if (Input.GetMouseButton(1))
            {
                _orbitRadius += Input.GetAxis("Mouse Y") * _zoomSpeed;
            }
            _orbitRadius -= Input.mouseScrollDelta.y * _zoomSpeed;
            _orbitRadius = Mathf.Clamp(_orbitRadius, 0.1f, 5);

            // Position camera to match new rotation values
            Quaternion rotation = Quaternion.Euler(_orbitX, _orbitY, 0);
            transform.position = (rotation * Vector3.up) * _orbitRadius;

            // Look at origin
            transform.LookAt(Vector3.zero);
        }
    }
}
