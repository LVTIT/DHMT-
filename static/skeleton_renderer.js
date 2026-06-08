import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createHumanoidMesh, updateHumanoidMesh } from './humanoid_mesh.js';

const SEGMENT_GROUPS = {
    torso: { pairs: [[11, 12], [11, 23], [12, 24], [23, 24]], color: 0xe9dfca },
    leftArm: { pairs: [[11, 13], [13, 15]], color: 0x50d2c2 },
    rightArm: { pairs: [[12, 14], [14, 16]], color: 0xff8068 },
    leftLeg: { pairs: [[23, 25], [25, 27], [27, 29], [27, 31]], color: 0x8bd45f },
    rightLeg: { pairs: [[24, 26], [26, 28], [28, 30], [28, 32]], color: 0xf4bd4f },
    face: { pairs: [[0, 11], [0, 12]], color: 0x94988e },
};

const SCALE = 1.65;
const JOINT_COUNT = 33;

export class SkeletonRenderer {
    constructor(container) {
        this.container = container;
        this.autoCenter = true;
        this.showSkeleton = true;
        this.showHumanoid = true;
        this.showIk = false;
        this.latestPoints = [];

        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x111315);
        this.scene.fog = new THREE.FogExp2(0x111315, 0.16);

        this.camera = new THREE.PerspectiveCamera(48, window.innerWidth / window.innerHeight, 0.01, 80);
        this.camera.position.set(0.38, 0.48, 2.55);

        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.08;
        container.appendChild(this.renderer.domElement);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.06;
        this.controls.minDistance = 0.55;
        this.controls.maxDistance = 7;
        this.controls.target.set(0, 0.2, 0);

        this.root = new THREE.Group();
        this.scene.add(this.root);

        this._initLights();
        this._initGround();
        this._initSkeleton();
        this._initIkOverlay();

        this.humanoid = createHumanoidMesh();
        this.root.add(this.humanoid);

        window.addEventListener('resize', () => this._resize());
        this._animate();
    }

    update(payload) {
        if (!payload || !payload.detected || !payload.landmarks) {
            this.latestPoints = [];
            this._setSkeletonVisible(false);
            updateHumanoidMesh(this.humanoid, [], false);
            this._setIkVisible(false);
            return;
        }

        const points = this._transformLandmarks(payload.landmarks);
        this.latestPoints = points;
        const visibility = payload.visibility || [];
        this._updateJoints(points, visibility);
        this._updateBones(points, visibility);
        this._updateIk(payload.metadata?.ik_demo || {});
        updateHumanoidMesh(this.humanoid, points, this.showHumanoid);
    }

    setOverlay(name, active) {
        if (name === 'grid') this.grid.visible = active;
        if (name === 'skeleton') {
            this.showSkeleton = active;
            this._setSkeletonVisible(active);
        }
        if (name === 'humanoid') {
            this.showHumanoid = active;
            this.humanoid.visible = active;
        }
        if (name === 'ik') {
            this.showIk = active;
            this._setIkVisible(active);
        }
        if (name === 'center') {
            this.autoCenter = active;
        }
    }

    resetCamera() {
        this.camera.position.set(0.38, 0.48, 2.55);
        this.controls.target.set(0, 0.2, 0);
        this.controls.update();
    }

    _initLights() {
        this.scene.add(new THREE.HemisphereLight(0xf6f0dc, 0x24272b, 2.1));
        const key = new THREE.DirectionalLight(0xffffff, 1.25);
        key.position.set(2.8, 4.2, 2.4);
        key.castShadow = true;
        this.scene.add(key);
        const rim = new THREE.PointLight(0xff8068, 1.2, 5);
        rim.position.set(-1.8, 0.6, 1.2);
        this.scene.add(rim);
        const teal = new THREE.PointLight(0x50d2c2, 0.8, 5);
        teal.position.set(1.4, 0.2, -1.8);
        this.scene.add(teal);
    }

    _initGround() {
        const ground = new THREE.Mesh(
            new THREE.PlaneGeometry(7, 7),
            new THREE.MeshStandardMaterial({ color: 0x17191c, roughness: 0.9 })
        );
        ground.rotation.x = -Math.PI / 2;
        ground.position.y = -1.05;
        ground.receiveShadow = true;
        this.scene.add(ground);
        this.grid = new THREE.GridHelper(7, 28, 0x50d2c2, 0x363a3f);
        this.grid.position.y = -1.045;
        this.scene.add(this.grid);
    }

    _initSkeleton() {
        this.jointGroup = new THREE.Group();
        this.lineGroup = new THREE.Group();
        this.root.add(this.lineGroup, this.jointGroup);

        const jointGeometry = new THREE.SphereGeometry(0.018, 16, 16);
        this.joints = [];
        for (let i = 0; i < JOINT_COUNT; i++) {
            const material = new THREE.MeshStandardMaterial({
                color: i < 11 ? 0xe9dfca : 0xf4bd4f,
                emissive: 0x201a05,
                roughness: 0.38,
            });
            const sphere = new THREE.Mesh(jointGeometry, material);
            sphere.castShadow = true;
            sphere.visible = false;
            this.jointGroup.add(sphere);
            this.joints.push(sphere);
        }

        this.lineSegments = [];
        for (const group of Object.values(SEGMENT_GROUPS)) {
            const positions = new Float32Array(group.pairs.length * 2 * 3);
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            const line = new THREE.LineSegments(
                geometry,
                new THREE.LineBasicMaterial({ color: group.color, linewidth: 2 })
            );
            line.visible = false;
            line.userData.pairs = group.pairs;
            this.lineGroup.add(line);
            this.lineSegments.push(line);
        }
    }

    _initIkOverlay() {
        this.ikGroup = new THREE.Group();
        this.root.add(this.ikGroup);
        this.ikLines = ['left_arm', 'right_arm'].map(() => {
            const geometry = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3(),
            ]);
            const line = new THREE.Line(
                geometry,
                new THREE.LineBasicMaterial({ color: 0xf4bd4f, linewidth: 2 })
            );
            line.visible = false;
            this.ikGroup.add(line);
            return line;
        });
    }

    _transformLandmarks(landmarks) {
        let center = [0, 0, 0];
        if (this.autoCenter && landmarks.length > 24) {
            center = [
                (landmarks[23][0] + landmarks[24][0]) * 0.5,
                (landmarks[23][1] + landmarks[24][1]) * 0.5,
                (landmarks[23][2] + landmarks[24][2]) * 0.5,
            ];
        }
        return landmarks.map((lm) => new THREE.Vector3(
            (lm[0] - center[0]) * SCALE,
            -(lm[1] - center[1]) * SCALE,
            -(lm[2] - center[2]) * SCALE
        ));
    }

    _updateJoints(points, visibility) {
        this.joints.forEach((joint, index) => {
            const ok = index < points.length && (visibility.length === 0 || visibility[index] >= 0.2);
            joint.userData.wasVisible = ok;
            joint.visible = ok && this.showSkeleton;
            if (ok) {
                joint.position.copy(points[index]);
                const confidence = visibility[index] ?? 1;
                joint.scale.setScalar(0.65 + confidence * 0.7);
            }
        });
    }

    _updateBones(points, visibility) {
        for (const line of this.lineSegments) {
            const array = line.geometry.attributes.position.array;
            let cursor = 0;
            let anyVisible = false;
            for (const [a, b] of line.userData.pairs) {
                const ok = a < points.length && b < points.length &&
                    (visibility.length === 0 || (visibility[a] >= 0.2 && visibility[b] >= 0.2));
                const pa = ok ? points[a] : new THREE.Vector3();
                const pb = ok ? points[b] : new THREE.Vector3();
                array[cursor++] = pa.x;
                array[cursor++] = pa.y;
                array[cursor++] = pa.z;
                array[cursor++] = pb.x;
                array[cursor++] = pb.y;
                array[cursor++] = pb.z;
                anyVisible = anyVisible || ok;
            }
            line.geometry.attributes.position.needsUpdate = true;
            line.visible = this.showSkeleton && anyVisible;
            line.userData.wasVisible = anyVisible;
        }
    }

    _updateIk(ikDemo) {
        const names = ['left_arm', 'right_arm'];
        names.forEach((name, index) => {
            const entry = ikDemo[name];
            const solved = entry?.solved || entry;
            if (!this.showIk || !Array.isArray(solved)) {
                this.ikLines[index].visible = false;
                this.ikLines[index].userData.wasVisible = false;
                return;
            }
            const points = this._transformLandmarks(solved);
            this.ikLines[index].geometry.setFromPoints(points);
            this.ikLines[index].visible = true;
            this.ikLines[index].userData.wasVisible = true;
        });
    }

    _setSkeletonVisible(visible) {
        this.joints.forEach((joint) => {
            joint.visible = visible && joint.userData.wasVisible;
        });
        this.lineSegments.forEach((line) => {
            line.visible = visible && line.userData.wasVisible;
        });
    }

    _setIkVisible(visible) {
        this.ikLines.forEach((line) => {
            line.visible = visible && line.userData.wasVisible;
        });
    }

    _resize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    _animate() {
        requestAnimationFrame(() => this._animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}
