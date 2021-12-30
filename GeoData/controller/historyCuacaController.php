<?php
    require_once 'controller/service/view.php';

    class historyCuacaController {
        public function start() {
            return View::createView('historyCuaca.php', []);
        }
    }
?>